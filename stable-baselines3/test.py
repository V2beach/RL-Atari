from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# We will use atari wrapper (it will downsample the image and convert it to gray scale).

# There already exists an environment generator that will make and wrap atari environments correctly.
env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

model = A2C("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# Download / Upload Trained Agent and Continue Training
from google.colab import files

model.save("a2c_pong")
files.download("a2c_pong.zip")

# Upload train agent from your local machine

files.upload()

!du -h a2c*

# Load the agent, and then you can continue training

trained_model = A2C.load("a2c_pong", verbose=1)
env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)
trained_model.set_env(env)

trained_model.learn(int(0.5e6))

trained_model.save("a2c_pong_2")
files.download("a2c_pong_2.zip")