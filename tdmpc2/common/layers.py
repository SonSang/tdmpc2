import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble

from t2a.design_opt.models.gnn import GNNSimple


class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		modules = nn.ModuleList(modules)
		fn, params, _ = combine_state_for_ensemble(modules)
		self.vmap = torch.vmap(fn, in_dims=(0, 0, None), randomness='different', **kwargs)
		self.params = nn.ParameterList([nn.Parameter(p) for p in params])
		self._repr = str(modules)

	def forward(self, *args, **kwargs):
		return self.vmap([p for p in self.params], (), *args, **kwargs)

	def __repr__(self):
		return 'Vectorized ' + self._repr


class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		padding = tuple([self.pad] * 4)
		assert h == w
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div_(255.).sub_(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""
	
	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim
	
	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)
		
	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=nn.Mish(inplace=True), **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))
	
	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
	layers = [
		# ShiftAug(), PixelPreprocess(),
		# @sanghyun: remove ShiftAug for now...
		PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=True),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=True),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=True),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)

def mconv(in_shape, num_channels, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 96 # assumes rgb observations to be 96x96
	layers = [
		# ShiftAug(), PixelPreprocess(),
		# @sanghyun: remove ShiftAug for now...
		PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=True),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=True),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=True),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
		elif k == 'rgb':
			out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)

def t2a_enc(cfg, env, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	Works for t2a environments.
	"""

	# rgb
	out['rgb'] = conv(cfg.obs_shape['rgb'], cfg.num_channels, act=SimNorm(cfg))
	rand_input = torch.randn((1, *cfg.obs_shape['rgb']), dtype=torch.float32)
	rand_output = out['rgb'](rand_input)
	rgb_out_dim = rand_output.shape[-1]
 
	# rgb_mlp: used to encode rgb features in gnn
	# use tanh for activation to match the range of node features
	node_obs_size = env._get_node_obs_size()
	out['rgb_mlp'] = mlp(rgb_out_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], node_obs_size, act=torch.nn.Tanh())
 
	# gnn
	node_obs_size = env._get_node_obs_size()
	gnn_input_size = node_obs_size
	gnn_spec = cfg['transform2act']['policy_specs']['control_gnn_specs']
	out['gnn'] = GNNSimple(in_dim=gnn_input_size, cfg=gnn_spec, node_dim=1)
 
	# gnn_mlp
	gnn_out_dim = out['gnn'].out_dim
	out['gnn_mlp'] = mlp(gnn_out_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))

	# final mlp for final latent
	out['final_mlp'] = mlp(cfg.latent_dim + cfg.latent_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))	
 
	return nn.ModuleDict(out)

def morphology_enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	Works for DMControl env with morphology information.
	"""
 
	# srgb
	out['srgb'] = mconv(cfg.obs_shape['srgb'], cfg.num_channels, act=SimNorm(cfg))
	rand_input = torch.randn((1, *cfg.obs_shape['srgb']), dtype=torch.float32)
	rand_output = out['srgb'](rand_input)
	rgb_out_dim = rand_output.shape[-1]
 
	NODE_FEATURE_SIZE = 64
 
	# srgb_mlp: used to encode srgb features in gnn or final output
	if cfg.morphology_use_gnn:
		out['srgb_mlp'] = mlp(rgb_out_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], NODE_FEATURE_SIZE, act=torch.nn.Tanh())
	else:
		out['srgb_mlp'] = mlp(rgb_out_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
  
	# node_mlp: used to define node feature, because
	# there could be multiple geoms in one node
	node_obs_size = 17		# @TODO: Fixed to 17 for now
	out['node_mlp'] = mlp(node_obs_size, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], NODE_FEATURE_SIZE, act=torch.nn.Tanh())
 
	# gnn
	gnn_input_size = NODE_FEATURE_SIZE
	gnn_spec = cfg['transform2act']['policy_specs']['control_gnn_specs']
	out['gnn'] = GNNSimple(in_dim=gnn_input_size, cfg=gnn_spec, node_dim=1)
 
	# gnn_mlp
	gnn_out_dim = out['gnn'].out_dim
	out['gnn_mlp'] = mlp(gnn_out_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))

	# final mlp for final latent
	out['final_mlp'] = mlp(rgb_out_dim + cfg.latent_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))	
 
	return nn.ModuleDict(out)