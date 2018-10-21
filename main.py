import gym

from master_model import MasterCNN
from target_model import TargetCNN

if __name__ == '__main__':
    ENV_NAME = 'VirtualDrone-v0'
    env = gym.make(ENV_NAME)  # environment initialization

    master_model = MasterCNN(env.action_space.n)
    target_model = TargetCNN(env.action_space.n)


