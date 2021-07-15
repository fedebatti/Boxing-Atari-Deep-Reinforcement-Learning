import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from value_model import value_model
from policy_model import policy_model
from obs_preprocessing import observation_preprocessing

class reinforce_agent():

    scores_file = 'scores_history.txt'

    def __init__(self, gamma = 0.99):
        self.gamma = gamma
        self.v_opt = tf.keras.optimizers.Adam(learning_rate=5e-6) #OLD VALUE 5e-6
        self.p_opt = tf.keras.optimizers.Adam(learning_rate=5e-6) #OLD VALUE 5e-6
        self.v_model = value_model()
        self.p_model = policy_model()
        self.prediction_model = policy_model.create_prediction_model()
        self.top_avg_reward = -1000
    
    def play_one_step(self, state):
       # actions_prob = self.p_model(np.array([state]))
        actions_prob = self.p_model(np.expand_dims(state,axis=0))
        #print(prob)
        prob = actions_prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])
        # action = np.random.choice([i for i in range(env.action_space.n)], 1, p=prob[0])
        # log_prob = tf.math.log(prob[0][action]).numpy()
        # self.log_prob = log_prob[0]
        # #print(self.log_prob)
        # return action[0]


    def policy_loss(self, actions_prob, action, delta):
        dist = tfp.distributions.Categorical(probs=actions_prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*delta
        return loss**2

    def training_step(self, state, action, G):
        with tf.GradientTape() as tape_v, tf.GradientTape() as tape_p:
            #Prediction for p
            p = self.p_model(np.expand_dims(state,axis=0))
            #Prediciton for v
            v =  self.v_model(np.expand_dims(state,axis=0))
            #Delta value
            delta = G - v
            p_loss = self.policy_loss(p, action, delta)
            v_loss = delta**2
        grads_p = tape_p.gradient(p_loss, self.p_model.trainable_variables)
        grads_v = tape_v.gradient(v_loss, self.v_model.trainable_variables)
        self.p_opt.apply_gradients(zip(grads_p, self.p_model.trainable_variables))
        self.v_opt.apply_gradients(zip(grads_v, self.v_model.trainable_variables))
        return p_loss, v_loss

    def training(self, steps_list, total_reward):
        G = total_reward
        v_loss_list = []
        p_loss_list = []
        for step in steps_list:
            step_index = step[0]
            state = step[1]
            action = step[2]
            reward = step[3]
            G = G - reward
            p_loss, v_loss = self.training_step(state, action, G)
            v_loss_list.append(v_loss)
            p_loss_list.append(p_loss)
            print(f"Step: {step_index}, V_loss = {v_loss}, P_loss = {p_loss}")
        return v_loss_list, p_loss_list
        
    def play_one_step_greedy(self, state, evaluation=False):
      if evaluation:
        actions_prob = self.p_model(np.expand_dims(state,axis=0))
        return np.argmax(actions_prob)
      else:
        actions_prob = self.prediction_model(np.expand_dims(state,axis=0))
        return np.argmax(actions_prob)

    def play_episode_evaluation(self, env):
        total_reward = 0
        state = env.reset()
        terminated = False
        while not terminated:
            state = observation_preprocessing(state)
            #agent_states = np.expand_dims(np.array(states).reshape((80,80,4)), axis=0)
            action = self.play_one_step_greedy(state, evaluation=True)
            # Take action
            state, reward, terminated, _ = env.step(action)
            total_reward += reward
        return total_reward

    def evaluate_model(self, env):
        num_episodes = 10
        reward_list = []
        #print("Playing with the last training's model")
        for ep in range(num_episodes):
            ep_reward = self.play_episode_evaluation(env)
            reward_list.append(ep_reward)
            print(f"Starting completed for episode {ep} with total reward: {ep_reward}")
        return np.mean(np.array(reward_list)) 

    def model_compare(self, env, episode):
        print("Evaluating training model")
        training_model_value = self.evaluate_model(env)
        with open(self.scores_file, "a") as scores_f:
            scores_f.write(f"episode:{episode} average_score:{training_model_value}\n")
        if training_model_value > self.top_avg_reward:
            print("New best model found")
            self.top_avg_reward = training_model_value
            return True
        else:
            print("New trained model performs worse than the previous best")
            return False