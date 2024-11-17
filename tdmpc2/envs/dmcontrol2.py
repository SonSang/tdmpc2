from envs.dmcontrol import make_env as make_dmcontrol_env
from envs.wrappers.mjgraph.mjgraph import MJGraphWrapper
from dm_control import suite
from dm_control.utils import xml_tools
from dm_control.suite.common import _SUITE_DIR

from lxml import etree
import numpy as np

'''
DMControl environments that satisfy following requirements:

1. The environment must have a get_model_and_assets function that returns the model XML string and a dict of assets.
2. The get_model_and_assets function must not have any arguments.
3. The Mujoco model is only comprised of articulated bodies and joints.
4. The worldboy has only one child body.
'''
from dm_control.suite.acrobot import get_model_and_assets as acrobot_get_model_and_assets
from dm_control.suite.cheetah import get_model_and_assets as cheetah_get_model_and_assets
from dm_control.suite.humanoid import get_model_and_assets as humanoid_get_model_and_assets
from dm_control.suite.hopper import get_model_and_assets as hopper_get_model_and_assets
from dm_control.suite.pendulum import get_model_and_assets as pendulum_get_model_and_assets
from dm_control.suite.reacher import get_model_and_assets as reacher_get_model_and_assets
from dm_control.suite.humanoid_CMU import get_model_and_assets as humanoid_CMU_get_model_and_assets
from dm_control.suite.walker import get_model_and_assets as walker_get_model_and_assets

DOMAIN = ('acrobot', 'cheetah', 'humanoid', 'hopper', 'pendulum', 'reacher', 'humanoid_CMU', 'walker')

'''
Functions to perturb morphology settings
'''
def perturb_model(model: str, domain: str, seed: int = None):
	
	assert domain in DOMAIN, f'Invalid domain: {domain}'
	
	parser = etree.XMLParser(remove_blank_text=True)
	mjcf = etree.XML(model, parser)

	if domain == 'humanoid':
		# thigh length
		curr_right_thigh_fromto = xml_tools.find_element(mjcf, 'geom', 'right_thigh').attrib["fromto"]
		curr_right_thigh_length = float(curr_right_thigh_fromto.split()[-1])
  
		curr_left_thigh_fromto = xml_tools.find_element(mjcf, 'geom', 'left_thigh').attrib["fromto"]
		curr_left_thigh_length = float(curr_left_thigh_fromto.split()[-1])
  
		assert curr_right_thigh_length == curr_left_thigh_length, 'Right and left thigh lengths are different'

		# shin pos
		curr_right_shin_pos = xml_tools.find_element(mjcf, 'body', 'right_shin').attrib["pos"]
		curr_left_shin_pos = xml_tools.find_element(mjcf, 'body', 'left_shin').attrib["pos"]
		curr_right_shin_pos = float(curr_right_shin_pos.split()[-1])
		curr_left_shin_pos = float(curr_left_shin_pos.split()[-1])
  
		assert curr_right_shin_pos == curr_left_shin_pos, 'Right and left shin positions are different'
  
		# shin length
		curr_right_shin_fromto = xml_tools.find_element(mjcf, 'geom', 'right_shin').attrib["fromto"]
		curr_right_shin_length = float(curr_right_shin_fromto.split()[-1])
  
		curr_left_shin_fromto = xml_tools.find_element(mjcf, 'geom', 'left_shin').attrib["fromto"]
		curr_left_shin_length = float(curr_left_shin_fromto.split()[-1])
  
		assert curr_right_shin_length == curr_left_shin_length, 'Right and left shin lengths are different'

		# foot pos
		curr_right_foot_pos = xml_tools.find_element(mjcf, 'body', 'right_foot').attrib["pos"]
		curr_left_foot_pos = xml_tools.find_element(mjcf, 'body', 'left_foot').attrib["pos"]
		curr_right_foot_pos = float(curr_right_foot_pos.split()[-1])
		curr_left_foot_pos = float(curr_left_foot_pos.split()[-1])
  
		# perturb thigh length
		right_thigh_to_shin_dist = curr_right_shin_pos - curr_right_thigh_length
		left_thigh_to_shin_dist = curr_left_shin_pos - curr_left_thigh_length
  
		rand_range = (0.7, 2.0)
		perturbed_right_thigh_length = curr_right_thigh_length * np.random.uniform(*rand_range)
		perturbed_left_thigh_length = perturbed_right_thigh_length
  
		perturbed_right_shin_pos = perturbed_right_thigh_length + right_thigh_to_shin_dist
		perturbed_left_shin_pos = perturbed_left_thigh_length + left_thigh_to_shin_dist
  
		print("[Morphology Change] Thigh length change from {} to {}".format(curr_right_thigh_length, perturbed_right_thigh_length))
  
		# perturb shin length
		right_shin_to_foot_dist = curr_right_foot_pos - curr_right_shin_length
		left_shin_to_foot_dist = curr_left_foot_pos - curr_left_shin_length
  
		perturbed_right_shin_length = curr_right_shin_length * np.random.uniform(*rand_range)
		perturbed_left_shin_length = perturbed_right_shin_length
  
		perturbed_right_foot_pos = perturbed_right_shin_length + right_shin_to_foot_dist
		perturbed_left_foot_pos = perturbed_left_shin_length + left_shin_to_foot_dist
  
		print("[Morphology Change] Shin length change from {} to {}".format(curr_right_shin_length, perturbed_right_shin_length))
  
		# update thigh length, shin pos
		xml_tools.find_element(mjcf, 'geom', 'right_thigh').attrib["fromto"] = f"0 0 0 0 .01 {perturbed_right_thigh_length}"
		xml_tools.find_element(mjcf, 'geom', 'left_thigh').attrib["fromto"] = f"0 0 0 0 -.01 {perturbed_left_thigh_length}"
		xml_tools.find_element(mjcf, 'body', 'right_shin').attrib["pos"] = f"0 .01 {perturbed_right_shin_pos}"
		xml_tools.find_element(mjcf, 'body', 'left_shin').attrib["pos"] = f"0 -.01 {perturbed_left_shin_pos}"
  
		# update shin length
		xml_tools.find_element(mjcf, 'geom', 'right_shin').attrib["fromto"] = f"0 0 0 0 0 {perturbed_right_shin_length}"
		xml_tools.find_element(mjcf, 'geom', 'left_shin').attrib["fromto"] = f"0 0 0 0 0 {perturbed_left_shin_length}"
		xml_tools.find_element(mjcf, 'body', 'right_foot').attrib["pos"] = f"0 0 {perturbed_right_foot_pos}"
		xml_tools.find_element(mjcf, 'body', 'left_foot').attrib["pos"] = f"0 0 {perturbed_left_foot_pos}"
 
	elif domain == "walker":
		
		'''
		Change leg length
  		'''
		# select random leg length
		default_leg_length = 0.25
		rand_range = (0.5, 2.0)
  
		if seed >= 0:
			rng = np.random.default_rng(seed=seed)
			leg_length = default_leg_length * rng.uniform(*rand_range)
		else:
			leg_length = default_leg_length * -seed
		leg_length_update = leg_length - default_leg_length
  
		'''
		Update parameters to match the new leg length
  		'''
    
		# torso
		torso = xml_tools.find_element(mjcf, 'body', 'torso')
		torso_pos = torso.attrib['pos']
		torso_pos_z = float(torso_pos.split()[-1])
		n_torso_pos_z = torso_pos_z + leg_length_update
		xml_tools.find_element(mjcf, 'body', 'torso').attrib['pos'] = f'0 0 {n_torso_pos_z}'
  
		# right & left leg
		right_leg = xml_tools.find_element(mjcf, 'body', 'right_leg')
		left_leg = xml_tools.find_element(mjcf, 'body', 'left_leg')
		right_leg_pos = right_leg.attrib['pos']
		left_leg_pos = left_leg.attrib['pos']
		right_leg_pos_z = float(right_leg_pos.split()[-1])
		left_leg_pos_z = float(left_leg_pos.split()[-1])
		n_right_leg_pos_z = right_leg_pos_z - leg_length_update
		n_left_leg_pos_z = left_leg_pos_z - leg_length_update
		xml_tools.find_element(mjcf, 'body', 'right_leg').attrib['pos'] = f'0 0 {n_right_leg_pos_z}'
		xml_tools.find_element(mjcf, 'body', 'left_leg').attrib['pos'] = f'0 0 {n_left_leg_pos_z}'
  
		# right & left knee joint
		xml_tools.find_element(mjcf, 'joint', 'right_knee').attrib['pos'] = f'0 0 {leg_length}'
		xml_tools.find_element(mjcf, 'joint', 'left_knee').attrib['pos'] = f'0 0 {leg_length}'
  
		# right & left leg geom
		xml_tools.find_element(mjcf, 'geom', 'right_leg').attrib['size'] = f'0.04 {leg_length}'
		xml_tools.find_element(mjcf, 'geom', 'left_leg').attrib['size'] = f'0.04 {leg_length}'
  
		# right & left foot
		xml_tools.find_element(mjcf, 'body', 'right_foot').attrib['pos'] = f'0.06 0 {-leg_length}'
		xml_tools.find_element(mjcf, 'body', 'left_foot').attrib['pos'] = f'0.06 0 {-leg_length}'

	else:
		raise NotImplementedError(f'Perturbation for domain {domain} is not implemented')
  	
	return etree.tostring(mjcf, pretty_print=True)
		

def make_env(cfg):
	'''
	Make DMControl env, and wrap it with a MJGraph wrapper.
	This wrapper forces the environment to give image observation
	and graph observation.
	'''
 
	if cfg.morphology:
     		
		# get original Mujoco model
		domain, _ = cfg.task.replace('-', '_').split('_', 1)
		if domain not in DOMAIN:
			raise ValueError(f'Invalid domain: {domain}')
		original_model, _ = globals()[f'{domain}_get_model_and_assets']()
	
		# backup original model
		with open(f'{_SUITE_DIR}/{domain}_backup.xml', 'w') as f:
			model_str = original_model.decode('utf-8')
			f.write(model_str)
 		
		model = perturb_model(original_model, domain, cfg.morphology_seed)
 
		# save perturbed morphology
		with open(f'{_SUITE_DIR}/{domain}.xml', 'w') as f:
			model_str = model.decode('utf-8')
			f.write(model_str)
	
	env0 = make_dmcontrol_env(cfg)
 
	# restore original model
	if cfg.morphology:
		with open(f'{_SUITE_DIR}/{domain}.xml', 'w') as f:
			model_str = original_model.decode('utf-8')
			f.write(model_str)
		env = MJGraphWrapper(model, env0)
	else:
		env = env0
	
	return env