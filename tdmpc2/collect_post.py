import os
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
from tqdm import tqdm
from tensordict import TensorDict

'''
Parse saved trajectories into trainable format
'''
dataset_path = '/home/sanghyun/Documents/cogrobot/tdmpc2/walker_dataset'
traj_paths = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
traj_paths = sorted(traj_paths)

save_dir = '/home/sanghyun/Documents/cogrobot/tdmpc2/walker_dataset/train/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

import zipfile
bar = tqdm(traj_paths)
for tpath in bar:
    episode_paths = [f for f in os.listdir(os.path.join(dataset_path, tpath, 'Traj')) if f.endswith('.zip')]
    episode_paths = sorted(episode_paths)
    
    final_episode = episode_paths[-1]
    
    zip_path = os.path.join(dataset_path, tpath, 'Traj', final_episode)
    unzip_path = os.path.join(dataset_path, tpath, 'Traj', final_episode.split('.')[0])
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_path)
        
    # prev obs
    prev_rgb_path = os.path.join(unzip_path, 'prev_rgb.npy')
    prev_rgb = np.load(prev_rgb_path)
    num_frame = len(prev_rgb)
    
    # remove unzip file
    import shutil
    shutil.rmtree(unzip_path)
    
    # save video
    video_path = os.path.join(dataset_path, tpath, 'final_episode.gif')
    from PIL import Image
    imgs = [Image.fromarray(img) for img in prev_rgb]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(video_path, save_all=True, append_images=imgs[1:999], duration=50, loop=0)
    
    # import numpy as np
    # import cv2
    # size = 720*16//9, 720
    # fps = 25
    
    # video_path = os.path.join(dataset_path, tpath, 'final_episode.mp4')
    # out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (96, 96))
    # for frame in range(num_frame):
    #     data = prev_rgb[frame]
    #     out.write(data)
    # out.release()