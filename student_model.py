from keras.layers import Conv2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam, SGD


class StudentCNN:
    def __init__(self, action_size, hidden_fc_size, learning_rate):
        self.action_size = action_size
        self.hidden_fc_size = hidden_fc_size
        self.lr = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='valid', activation='relu',
                         data_format='channels_first', input_shape=(1, 200, 200)))
        model.add(Conv2D(8, (1, 1), strides=(1, 1), padding='valid', activation='relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='valid', activation='relu'))
        model.add(Conv2D(16, (1, 1), strides=(1, 1), padding='valid', activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))

        # make convolution layers falttend (1 dimensional)
        model.add(Flatten())

        model.add(Dense(self.hidden_fc_size, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))

        sgd = SGD(lr=self.lr)
        model.compile(optimizer=sgd,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model
