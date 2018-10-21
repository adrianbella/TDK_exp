import gym

from master_model import MasterCNN
from student_model import StudentCNN
from logger import Logger

from agent import Agent

if __name__ == '__main__':

    ENV_NAME = 'VirtualDrone-v0'
    AGENT_TYPE = 'DQNAgent'
    database_limit = 992
    hidden_fc_size = 256
    env = gym.make(ENV_NAME)  # environment initialization
    file_path = './master_weights/' + ENV_NAME + '_' + AGENT_TYPE + '.h5f'

    master_model = MasterCNN(env.action_space.n)
    student_model = StudentCNN(env.action_space.n, hidden_fc_size)

    logger = Logger(AGENT_TYPE, ENV_NAME, hidden_fc_size, student_model)

    agent = Agent(env.action_space.n, database_limit, ENV_NAME, AGENT_TYPE, hidden_fc_size)

    try:
        master_model.model.load_weights(filepath=file_path)
        print('Loaded master_weights was successful')
    except ImportError:
        print('Loaded master_weights aborted! File not found:{} '.format(file_path))

    agent.fit_student(master_model, student_model, env, logger)

    print('end')
