from t2a.design_opt.envs import env_dict
import numpy as np

'''
Config class for Transform2Act.
'''
class Config:

    def __init__(self, cfg_dict):
        
        assert cfg_dict is not None, "cfg_dict is None"
        
        cfg = cfg_dict
        
        # training config
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.95)
        self.agent_specs = cfg.get('agent_specs', dict())
        self.policy_specs = cfg.get('policy_specs', dict())
        self.policy_optimizer = cfg.get('policy_optimizer', 'Adam')
        self.policy_lr = cfg.get('policy_lr', 5e-5)
        self.policy_momentum = cfg.get('policy_momentum', 0.0)
        self.policy_weightdecay = cfg.get('policy_weightdecay', 0.0)
        self.value_specs = cfg.get('value_specs', dict())
        self.value_optimizer = cfg.get('value_optimizer', 'Adam')
        self.value_lr = cfg.get('value_lr', 3e-4)
        self.value_momentum = cfg.get('value_momentum', 0.0)
        self.value_weightdecay = cfg.get('value_weightdecay', 0.0)
        self.adv_clip = cfg.get('adv_clip', np.inf)
        self.clip_epsilon = cfg.get('clip_epsilon', 0.2)
        self.num_optim_epoch = cfg.get('num_optim_epoch', 10)
        self.min_batch_size = cfg.get('min_batch_size', 50000)
        self.mini_batch_size = cfg.get('mini_batch_size', self.min_batch_size)
        self.eval_batch_size = cfg.get('eval_batch_size', 10000)
        self.max_epoch_num = cfg.get('max_epoch_num', 1000)
        self.seed = cfg.get('seed', 1)
        self.seed_method = cfg.get('seed_method', 'deep')
        self.save_model_interval = cfg.get('save_model_interval', 100)

        # anneal parameters
        self.scheduled_params = cfg.get('scheduled_params', dict())

        # env
        self.env_name = cfg.get('env_name', 'hopper')
        self.done_condition = cfg.get('done_condition', dict())
        self.env_specs = cfg.get('env_specs', dict())
        self.reward_specs = cfg.get('reward_specs', dict())
        self.obs_specs = cfg.get('obs_specs', dict())
        self.add_body_condition = cfg.get('add_body_condition', dict())
        self.max_body_depth = cfg.get('max_body_depth', 4)
        self.min_body_depth = cfg.get('min_body_depth', 1)
        self.enable_remove = cfg.get('enable_remove', True)
        self.skel_transform_nsteps = cfg.get('skel_transform_nsteps', 5)
        self.env_init_height = cfg.get('env_init_height', False)

        # robot config
        self.robot_param_scale = cfg.get('robot_param_scale', 0.1)
        self.robot_cfg = cfg.get('robot', dict())
      
def make_env(cfg):
    '''
    Make a Transform2Act environment.
    '''
    try:
        task_name = cfg.task
        task_env = task_name.split('_')[0]
        if task_env != "t2a":
            raise ValueError(f"Invalid task name: {task_name}")
    except:
        raise ValueError("Task name not provided")
    
    t2a_cfg = cfg.transform2act
    t2a_cfg = Config(t2a_cfg)
    
    env_name = t2a_cfg.env_name
    env_class = env_dict[env_name]
    env = env_class(t2a_cfg, None)
    
    return env