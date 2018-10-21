import logging
import datetime
import os
import keras
import csv


class Logger(keras.callbacks.Callback):
    def __init__(self, AGENT_TYPE, ENV_NAME, hidden_fc_size, student_model):
        self.agent = AGENT_TYPE
        self.ENV_NAME = ENV_NAME
        self.hidden_fc_size = hidden_fc_size
        self.directory = './log/'
        weights = './student_weights/'

        if not os.path.exists(weights):
            os.makedirs(weights)

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        logging.basicConfig(
            filename=self.directory + self.ENV_NAME + '_' + self.agent + '_' + str(self.hidden_fc_size) + '.log',

            level=logging.DEBUG)

        student_model.model.summary(print_fn=logging.info)

        with open(self.directory + self.ENV_NAME + '_' + self.agent + '_' + str(self.hidden_fc_size) + '.csv',
                  'w+') as csvfile:
            fieldnames = ['acc', 'batch', 'loss', 'size']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    def on_train_begin(self, logs={}):
        self.acc = []
        self.batch = []
        self.loss = []
        self.size = []
        print('training_begin')

    def on_batch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.batch.append(logs.get('batch'))
        self.loss.append(logs.get('loss'))
        self.size.append(logs.get('size'))

    def on_train_end(self, logs={}):
        if len(self.acc) > 1:
            with open(self.directory + self.ENV_NAME + '_' + self.agent + '_' + str(self.hidden_fc_size) + '.csv',
                      'a+') as csvfile:
                fieldnames = ['acc', 'batch', 'loss', 'size']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for i in range(len(self.acc)):
                    writer.writerow({'acc': self.acc[i], 'batch': self.batch[i], 'loss': self.loss[i],
                                     'size': self.size[i]})

        self.acc.clear()
        self.batch.clear()
        self.loss.clear()
        self.size.clear()
        print('training_end')
