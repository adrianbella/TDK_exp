import gym

from master_model import MasterCNN
from student_model import StudentCNN
from logger import Logger

from agent import Agent

if __name__ == '__main__':

    ENV_NAME = 'VirtualDrone-v0'
    AGENT_TYPE = 'DQNAgent'
    database_limit = 1024
    hidden_fc_size = 256
    learning_rate = 0.0005
    env = gym.make(ENV_NAME)  # environment initialization
    file_path = './master_weights/' + ENV_NAME + '_' + AGENT_TYPE + '.h5f'

    master_model = MasterCNN(env.action_space.n, file_path)
    student_model = StudentCNN(env.action_space.n, hidden_fc_size, learning_rate)

    logger = Logger(AGENT_TYPE, ENV_NAME, hidden_fc_size, student_model)

    agent = Agent(env.action_space.n, database_limit, ENV_NAME, AGENT_TYPE, hidden_fc_size)

    agent.fit_student(master_model, student_model, env, logger)

    print('end')
