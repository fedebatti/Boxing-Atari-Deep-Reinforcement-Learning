import imageio
import base64
import IPython
from reinforce_agent import reinforce_agent
from obs_preprocessing import observation_preprocessing
import gym
import numpy as np
class pacman_video:

    def __init__(self, action_space_dim, policy_path, first_state):
        self.agent = reinforce_agent(action_space_dim)
        print("Loading best model weights")
        self.agent.prediction_model(np.expand_dims(first_state,axis=0))
        self.agent.prediction_model.load_weights(policy_path)
    # create_video(env, training_network)


    #Generate self play video to evaluate the performances of the system
    def create_video(self, env, video_filename = 'ms_pacman'):
        def embed_mp4(filename):
            """Embeds an mp4 file in the notebook."""
            video = open(filename,'rb').read()
            b64 = base64.b64encode(video)
            tag = '''
            <video width="640" height="480" controls>
            <source src="data:video/mp4;base64,{0}" type="video/mp4">
            Your browser does not support the video tag.
            </video>'''.format(b64.decode())

            return IPython.display.HTML(tag)


        num_episodes = 10
        print("Setting video name")
        video_filename = video_filename + ".mp4"
        with imageio.get_writer(video_filename, fps=60) as video:
            for ep in range(num_episodes):
                print("Starting simulation for episode {}".format(ep))
                states = []
                state = env.reset()
                terminated = False
                while not terminated:
                    state = observation_preprocessing(state)
                    states.append(state)
                    #agent_states = np.expand_dims(np.array(states).reshape((80,80,4)), axis=0)
                    action = self.agent.play_one_step_greedy(state)
                    # Take action
                    state, _, terminated, _ = env.step(action)
                    states.append(state)
                    video.append_data(state)
        print("Simulation completed video is ready to be downloaded")
        embed_mp4(video_filename)