import keras

import numpy as np


class Agent:
    def __init__(self, action_size, database_limit, ENV_NAME, AGENT_TYPE, hidden_fc_size):
        self.action_size = action_size
        self.database_limit = database_limit
        self.ENV_NAME = ENV_NAME
        self.AGENT_TYPE = AGENT_TYPE
        self.hidden_fc_size = hidden_fc_size

    def fit_student(self, master, student, env, logger):

        labels = np.zeros((self.database_limit, 1), dtype=np.int32)
        observations = np.zeros((self.database_limit, 1, 200, 200), dtype=np.int32)

        for i in range(0, self.database_limit):
            observation = env.reset()
            observation = np.expand_dims(np.expand_dims(observation, axis=0), axis=0)

            q_values = master.model.predict(observation, batch_size=1)

            labels[i] = np.argmax(q_values, axis=1)
            observations[i][0] = observation

        for j in range(0, 1000):

            one_hot_labels = keras.utils.to_categorical(labels, self.action_size)
            student.model.fit(observations, one_hot_labels, epochs=2, batch_size=32, callbacks=[logger], verbose=2)
            student.model.save_weights(
                filepath='./student_weights/' + self.ENV_NAME + '_' + self.AGENT_TYPE + '_' + str(
                    self.hidden_fc_size) + '.h5f',
                overwrite=True)
