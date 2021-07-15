import gym
from weights_manager import weights_manager
from video_simulation import pacman_video
from pathlib import Path
from obs_preprocessing import observation_preprocessing


policy_path = "policy-weights.h5"
env = gym.make("Boxing-v0")
first_state = observation_preprocessing(env.reset())
action_space_dim = len(env.env.get_action_meanings())
policy_weights_exist = Path(policy_path).exists() and Path(policy_path).is_file()
if policy_weights_exist:
    print("Pretrained model found starting simulation...")
    video = pacman_video(action_space_dim, policy_path, first_state)
    video.create_video(env)
else:
    print("Pretrained model not found!")