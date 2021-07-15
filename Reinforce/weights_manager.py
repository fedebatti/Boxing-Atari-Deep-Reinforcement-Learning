from pathlib import Path
import numpy as np
from reinforce_agent import reinforce_agent

class weights_manager:

    def __init__(self, value_path_prefix, policy_path_prefix, path_suffix):
        self.value_path_prefix = value_path_prefix
        self.policy_path_prefix = policy_path_prefix
        self.path_suffix = path_suffix
        self.value_path = value_path_prefix + path_suffix
        self.policy_path = policy_path_prefix + path_suffix


    def check_pretrained_weights(self, agent):
        print("Checking for pretrained model weights")
        value_weights_exist = Path(self.value_path).exists() and Path(self.value_path).is_file()
        policy_weights_exist = Path(self.policy_path).exists() and Path(self.policy_path).is_file()
        if value_weights_exist and policy_weights_exist:
            print("Existing weights file found, loading files on the models")
            dummy_input = np.zeros((1, 88, 80, 1))
            agent.v_model(dummy_input)
            agent.p_model(dummy_input)
            agent.v_model.load_weights(self.value_path)
            agent.p_model.load_weights(self.policy_path)
            print("Successfully loaded weights")
            return True
        else:
            print("Pretrained models not found starting training from the beginning")
            return False

    def update_weights(self, agent, episode):
        print("Saving new best model weights")
        new_value_path = self.value_path_prefix + str(episode) + self.value_path_suffix
        new_policy_path = self.policy_path_prefix + str(episode) + self.policy_path_suffix
        agent.v_model.save_weights(new_value_path)
        agent.p_model.save_weights(new_policy_path)

