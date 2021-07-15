"MS Pacman 2.0 Reinforce.py" is the file that contains the main (even if now the game is Boxing),
"obs_preprocessing.py" is deprecated,
"montecarlo.py"  contains the function to make a full rollout on an episode,
"weights_manager.py" manages models' weights save and load,
"policy/value_model.py" contain the models,
"reinforce_agent.py" manages the agent containing training and step functions, but also model evaluation function,
"video_simulation.py" contains the functions necessary to simulate a greedy play of 5 episodes with the model weights loaded,
"simulation.py" is the main file to execute for the simulation, it requires a policy weights file to load before executing, it
                                  generates a video.mp4 contaning the agent playing 10 episodes.

