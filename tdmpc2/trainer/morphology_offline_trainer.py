from trainer.offline_trainer import OfflineTrainer
import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from common.buffer import Buffer
from tensordict import TensorDict

def to_td(obs, action_shape, action=None, reward=None):
    """Creates a TensorDict for a new episode."""
    if isinstance(obs, dict):
        obs = TensorDict(obs, batch_size=(), device='cpu')

        # @sanghyun: make tensors have the leading dimension of 1
        for key in obs.keys():
            obs[key] = obs[key].unsqueeze(0).cpu()
    else:
        obs = obs.unsqueeze(0).cpu()
    if action is None:
        action = torch.full(action_shape, float('nan'))
    if reward is None:
        reward = torch.tensor(float('nan'))
    td = TensorDict(dict(
        obs=obs,
        action=action.unsqueeze(0),
        reward=reward.unsqueeze(0),
    ), batch_size=(1,))
    return td


class MorpOfflineTrainer(OfflineTrainer):
	
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()
		self._step = 0
  
	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
		)

	def train(self):
		"""Train a TD-MPC2 agent."""
		assert not self.cfg.multitask and self.cfg.task in {'walker_walk',} and self.cfg.morphology, \
			'Morphology Offline training only supports walker-walk.'

		# Load data
		agents = [f for f in os.listdir(self.cfg.data_dir) if os.path.isdir(os.path.join(self.cfg.data_dir, f))]
		agents = sorted(agents)
  
		agents_episodes = {}
  
		for agent in agents:
			agent_path = os.path.join(self.cfg.data_dir, agent)
			episode_paths = [f for f in os.listdir(os.path.join(agent_path, 'Traj')) if f.endswith('.zip')]
			episode_paths = sorted(episode_paths)
	
			agents_episodes[agent] = []
			for epath in episode_paths:
				full_epath = os.path.join(agent_path, 'Traj', epath)
				agents_episodes[agent].append(full_epath)
	
		num_optimization_steps = self.cfg.steps
		episode_length = 5000
		buffer_size = episode_length * 10		# 10 episodes per epoch
		num_steps_per_epoch = 10000
		num_epochs = (num_optimization_steps // num_steps_per_epoch) + 1
  
		for epoch in range(num_epochs):
			print(f'=== Epoch {epoch+1}/{num_epochs}')
   
			# Create buffer for sampling
			_cfg = deepcopy(self.cfg)
			_cfg.episode_length = episode_length
			_cfg.buffer_size = buffer_size
			_cfg.steps = buffer_size
			self.buffer = Buffer(_cfg)
   
			# randomly select 2 agents
			selected_agents = np.random.choice(agents, 2, replace=False)
   
			# for each agent, load 5 episodes
			for agent in selected_agents:
				episodes = agents_episodes[agent]
				permuted_episodes = np.random.permutation(len(episodes))
				
				cnt = 0
				for pei in permuted_episodes:
					epath = episodes[pei]
					try:
						unzip_path = epath.split('.')[0]
						import zipfile
						with zipfile.ZipFile(epath, 'r') as zip_ref:
							zip_ref.extractall(unzip_path)
       
						# prev obs
						prev_obs_srgb_path = os.path.join(unzip_path, 'prev_obs_srgb.npy')
						prev_obs_node_path = os.path.join(unzip_path, 'prev_obs_node.npy')
						prev_obs_edge_path = os.path.join(unzip_path, 'prev_obs_edge.npy')
						
						prev_obs_srgb = torch.tensor(np.load(prev_obs_srgb_path))
						prev_obs_node = torch.tensor(np.load(prev_obs_node_path))
						prev_obs_edge = torch.tensor(np.load(prev_obs_edge_path))
						
						# action
						action_path = os.path.join(unzip_path, 'action.npy')
						
						action = torch.tensor(np.load(action_path))
						action_shape = action[0].shape
						
						# reward
						reward_path = os.path.join(unzip_path, 'reward.npy')
						
						reward = torch.tensor(np.load(reward_path))
      
						# change to tds
						num_steps = len(prev_obs_srgb)
						tds = []
						for i in range(num_steps):
							obs = {
								'srgb': prev_obs_srgb[i],
								'node': prev_obs_node[i],
								'edge': prev_obs_edge[i],
							}
							if i == 0:
								td = to_td(obs, action_shape, None, None)
							else:
								prev_action = action[i-1]
								prev_reward = reward[i-1]
								td = to_td(obs, action_shape, prev_action, prev_reward)
							tds.append(td)
						tds = torch.cat(tds)
						# assert tds.shape[0] == episode_length, f'Expected episode length {tds.shape[0]} to match config episode length {episode_length}, please double-check your config.'
						self.buffer.add(tds)

						# remove extracted files
						import shutil
						shutil.rmtree(unzip_path)
      
						cnt += 1
						if cnt == 5:
							break
					except:
						continue
						
			print(f'Training agent for {num_steps_per_epoch} iterations...')
			metrics = {}
			bar = tqdm(range(num_steps_per_epoch))
			for i in bar:

				# Update agent
				train_metrics = self.agent.update(self.buffer)

				# Evaluate agent periodically
				
				if self._step % self.cfg.eval_freq == 0 or self._step % 100 == 0:
					metrics = {
						'iteration': self._step,
						'total_time': time() - self._start_time,
					}
					eval_metrics = {
						'step': self._step,
						'total_time': time() - self._start_time,
					}
					metrics.update(train_metrics)
					if self._step % self.cfg.eval_freq == 0:
						eval_metrics.update(self.eval())
						self.logger.log(eval_metrics, 'eval')
						if self._step > 0:
							self.logger.save_agent(self.agent, identifier=f'{self._step}')
							print(f'Saved agent at iteration {self._step}')
					self.logger.log(metrics, 'pretrain')
				
				self._step += 1
				
		self.logger.finish(self.agent)