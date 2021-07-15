import gym
from obs_preprocessing import observation_preprocessing
from reinforce_agent import reinforce_agent

#MonteCarlo Rollout implementation to play the full episode
def montecarlo_rollout(agent, env, training=True):
    #Variables init
    steps_list = []
    reward_accumulator = 0
    step_index = 0
    done = False
    #Env init
    state = env.reset()
    if training:
      #Playing each step until the end of the episode to train the network
      while not done:
        previous_state = observation_preprocessing(state)
        action = agent.play_one_step(previous_state)
        # Take action
        state, reward, done, info = env.step(action)
        # print(reward)
        reward_accumulator += reward
        steps_list.append((step_index, previous_state, action, reward))
        step_index += 1
      print("Total episode reward: {}".format(reward_accumulator))
      return steps_list, reward_accumulator
    else:
      #Playing each step until the end of the episode for simulation
      while not done:
        previous_state = observation_preprocessing(state)
        action = agent.play_one_step_greedy(previous_state)
        # Take action
        state, reward, done, _ = env.step(action)
        reward_accumulator += reward
        steps_list.append((step_index, previous_state, action, reward))
        step_index += 1
      print("Total episode reward: {}".format(reward_accumulator))
      return steps_list, reward_accumulator