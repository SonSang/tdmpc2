from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from common import layers, math, init


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self._action_masks = torch.zeros(len(cfg.tasks), cfg.action_dim)
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
		self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
		self.log_std_min = torch.tensor(cfg.log_std_min)
		self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
		
	def to(self, *args, **kwargs):
		"""
		Overriding `to` method to also move additional tensors to device.
		"""
		super().to(*args, **kwargs)
		if self.cfg.multitask:
			self._action_masks = self._action_masks.to(*args, **kwargs)
		self.log_std_min = self.log_std_min.to(*args, **kwargs)
		self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
		return self
	
	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def track_q_grad(self, mode=True):
		"""
		Enables/disables gradient tracking of Q-networks.
		Avoids unnecessary computation during policy optimization.
		This method also enables/disables gradients for task embeddings.
		"""
		for p in self._Qs.parameters():
			p.requires_grad_(mode)
		if self.cfg.multitask:
			for p in self._task_emb.parameters():
				p.requires_grad_(mode)

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		with torch.no_grad():
			for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
				p_target.data.lerp_(p.data, self.cfg.tau)
	
	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)
	
	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mu, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mu)

		if self.cfg.multitask: # Mask out unused action dimensions
			mu = mu * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
		pi = mu + eps * log_std.exp()
		mu, pi, log_pi = math.squash(mu, pi, log_pi)

		return mu, pi, log_pi, log_std

	def Q(self, z, a, task, return_type='min', target=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)
			
		z = torch.cat([z, a], dim=-1)
		out = (self._target_Qs if target else self._Qs)(z)

		if return_type == 'all':
			return out

		Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
		Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
		return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2

class T2AWorldModel(WorldModel):
	
	def __init__(self, cfg, env):
		
		nn.Module.__init__(self)
		
		self.cfg = cfg
		
		self._encoder = layers.t2a_enc(cfg, env)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
		self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
		self.log_std_min = torch.tensor(cfg.log_std_min)
		self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min
		
	def encode(self, obs, task):
		# encode rgb
		rgb_obs = obs['rgb']
		rgb_latent = self._encoder['rgb'](rgb_obs.to(dtype=torch.float32))
		
		# encode rgb to rgb node in gnn
		rgb_node_latent = self._encoder['rgb_mlp'](rgb_latent)
		rgb_node_latent = rgb_node_latent.unsqueeze(1)
		
		# encode robot
		node_obs = obs['node']
		edge_obs = obs['edge']
		device = node_obs.device
		
		# concatenate [rgb_node_latent] to [node_obs]
		e_node_obs = torch.cat([node_obs, rgb_node_latent], dim=1)
		
		# connect [rgb_node] to every other node
		num_nodes = node_obs.shape[1]
		n_edge_obs = []
		for i in range(num_nodes):
			n_edge_obs.append([i, num_nodes])
			n_edge_obs.append([num_nodes, i])
		
		n_edge_obs = torch.tensor(n_edge_obs, device=device).t().to(dtype=torch.long)
		n_edge_obs = n_edge_obs.unsqueeze(0).expand(node_obs.shape[0], -1, -1)
		e_edge_obs = torch.cat([edge_obs, n_edge_obs], dim=2)
			
		# encode robot node in gnn
		robot_latent = self._encoder['gnn'](e_node_obs, e_edge_obs[0])
		
		# additional mlp for robot latent
		robot_latent = self._encoder['gnn_mlp'](robot_latent)
		
		# concatenate [robot_latent] to [rgb_latent]
		prefinal_latent = torch.cat([rgb_latent, robot_latent], dim=-1)
		
		# final ml
		latent = self._encoder['final_mlp'](prefinal_latent)
		
		return latent

class MorphologyWorldModel(WorldModel):
	
	def __init__(self, cfg, use_gnn): # env):
		
		nn.Module.__init__(self)
		
		self.cfg = cfg
		self.use_gnn = use_gnn
		# self.env = env
		
		self._encoder = layers.morphology_enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
		self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
		self.log_std_min = torch.tensor(cfg.log_std_min)
		self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min
		
	def encode(self, obs, task):
		# encode srgb
		rgb_obs = obs['srgb']
		rgb_latent = self._encoder['srgb'](rgb_obs.to(dtype=torch.float32))
  
		if self.use_gnn:
			# encode rgb to rgb node in gnn
			rgb_node_latent = self._encoder['srgb_mlp'](rgb_latent)
			rgb_node_latent = rgb_node_latent.unsqueeze(1)
		else:
			# get final latent
			latent = self._encoder['srgb_mlp'](rgb_latent)
			return latent
		
		# encode robot
		node_obs = obs['node']
		edge_obs = obs['edge']
		edge_obs = edge_obs[:, :, 0, :2]	# select only connectivity information
		edge_obs = edge_obs.transpose(1, 2).to(dtype=torch.long)
		device = node_obs.device
  
		# extract per-node information using node_mlp
		node_feat_0 = self._encoder['node_mlp'](node_obs)
		node_feat_avg = node_feat_0.mean(dim=2)
		
		# concatenate [rgb_node_latent] to [node_obs]
		e_node_obs = torch.cat([node_feat_avg, rgb_node_latent], dim=1)
		
		# connect [rgb_node] to every other node
		num_nodes = node_obs.shape[1]
		n_edge_obs = []
		for i in range(num_nodes):
			n_edge_obs.append([i, num_nodes])
			n_edge_obs.append([num_nodes, i])
		
		n_edge_obs = torch.tensor(n_edge_obs, device=device).t().to(dtype=torch.long)
		n_edge_obs = n_edge_obs.unsqueeze(0).expand(node_obs.shape[0], -1, -1)
		e_edge_obs = torch.cat([edge_obs, n_edge_obs], dim=2)
			
		# encode robot node in gnn
		robot_latent = self._encoder['gnn'](e_node_obs, e_edge_obs[0])
		
		# additional mlp for robot latent
		robot_latent = self._encoder['gnn_mlp'](robot_latent)
		
		# concatenate [robot_latent] to [rgb_latent]
		prefinal_latent = torch.cat([rgb_latent, robot_latent], dim=-1)
		
		# final ml
		latent = self._encoder['final_mlp'](prefinal_latent)
		
		return latent