# IQL Environment Recovery Guide

Goal:
Recover a fully working Offline RL environment for CORL + IQL on a fresh Linux GPU server.

Repository already exists at:

```bash
/code/CORL
```

DO NOT reclone the repository.

---

# Step 1 — Create Conda Environment

```bash
conda create -n iql python=3.8 -y
conda activate iql
```

---

# Step 2 — Install Python Dependencies

```bash
cd /code/CORL
```

Backup requirements:

```bash
cp requirements/requirements_dev.txt requirements/requirements_dev.bak.txt
```

Replace GitHub D4RL dependency with pip version:

```bash
sed -i 's|git+https://github.com/tinkoff-ai/d4rl.*|d4rl==1.1|' requirements/requirements_dev.txt
```

Install:

```bash
pip install -r requirements/requirements_dev.txt
```

---

# Step 3 — Install MuJoCo 2.1

```bash
mkdir -p ~/.mujoco
cd ~/.mujoco
```

Download:

```bash
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
```

Extract:

```bash
tar -xvf mujoco210-linux-x86_64.tar.gz
```

---

# Step 4 — Configure Environment Variables

```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc

echo 'export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210' >> ~/.bashrc

echo 'export MUJOCO_GL=osmesa' >> ~/.bashrc

echo 'export PYOPENGL_PLATFORM=osmesa' >> ~/.bashrc

echo 'export D4RL_SUPPRESS_IMPORT_ERROR=1' >> ~/.bashrc

echo 'export WANDB_MODE=offline' >> ~/.bashrc

source ~/.bashrc
```

---

# Step 5 — Install Linux Dependencies

```bash
apt update

apt install -y \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    patchelf \
    libosmesa6-dev \
    xpra \
    xserver-xorg-dev
```

---

# Step 6 — Fix Cython Compatibility

```bash
pip uninstall -y Cython

pip install Cython==0.29.37
```

---

# Step 7 — Install mujoco-py

```bash
pip uninstall -y mujoco-py

pip install mujoco-py==2.1.2.14
```

Clean cache:

```bash
rm -rf ~/.cache/mujoco_py
```

IMPORTANT:
DO NOT delete:

```text
site-packages/mujoco_py/generated
```

---

# Step 8 — Install MJRL

```bash
pip install git+https://github.com/aravindr93/mjrl.git
```

---

# Step 9 — Verify GPU

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected:

* CUDA available
* NVIDIA GPU detected

---

# Step 10 — Verify MuJoCo

```bash
python -c "import mujoco_py; print('mujoco ok')"
```

Expected:

```text
mujoco ok
```

---

# Step 11 — Verify D4RL

```bash
python -c "import gym, d4rl; env=gym.make('halfcheetah-medium-v2'); print('env ok')"
```

Expected:

```text
env ok
```

Warnings about:

* gym deprecated
* flow missing
* carla missing

can be ignored.

---

# Step 12 — Run IQL Smoke Test

```bash
cd /code/CORL

python algorithms/offline/iql.py \
  --env halfcheetah-medium-v2 \
  --device cuda \
  --seed 0 \
  --max_timesteps 10000 \
  --eval_freq 5000
```

IMPORTANT:
Use parameter:

```bash
--env
```

NOT:

```bash
--env_name
```

---

# Step 13 — D4RL Dataset

Dataset may download slowly.

Expected location:

```bash
/root/.d4rl/datasets/
```

Approximate size:

```text
halfcheetah-medium-v2 ≈ 150MB
```

If automatic download stalls:

```bash
mkdir -p /root/.d4rl/datasets

wget -c \
  http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5 \
  -O /root/.d4rl/datasets/halfcheetah_medium-v2.hdf5
```

---

# Final Goal

Successfully run:

```text
IQL training loop
```

and observe logs like:

```text
eval/normalized_score
critic_loss
value_loss
actor_loss
```

If errors occur:

* automatically debug them
* prefer minimal fixes
* preserve CORL repository structure
* do not reclone CORL
