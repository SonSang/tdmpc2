# Collecting Trajectory

If not cloned this repository recursively, initialize submodule for SB3 support.

```bash
git submodule init
git submodule update
```

Then, go to `external/traj_collect`, and run following command to install SB3.

```bash
pip install -e .
```

Finally, run following commands to test trajectory collection. See `tdmpc2/collect.py` for more details.

```bash
python tdmpc2/collect.py task=walker_walk sb3_algo=sac steps=1000 morphology=True morphology_seed=1 ckpt_step=200      # DMControl Env
```

Find the saved trajectory and the parsed version of it at the `outputs` directory.

# Training Morphology Walker-walk

Online Training:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python tdmpc2/train.py task=walker_walk morphology=True morphology_use_gnn=True morphology_seed=100 steps=7000000 eval_episodes=4 buffer_size=50000 disable_wandb=False wandb_project=sanghyun_son wandb_entity=shh1295
```

Offline Training:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python tdmpc2/train.py task=walker_walk model_size=5 batch_size=1024 data_dir=/home/sanghyun/Documents/cogrobot/tdmpc2/walker_dataset eval_episodes=4 eval_freq=2000 morphology=True morphology_use_gnn=False morphology_seed=-1 disable_wandb=False wandb_project=sanghyun_son wandb_entity=shh1295
```