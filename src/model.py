from tensorflow.keras.layers import (Dense,
                                     Conv2D,
                                     MaxPooling2D,
                                     Dropout,
                                     Flatten
                                     )
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow_estimator.python.estimator import early_stopping

from src.dataloader import load_data


class CNN_Model(object):
    def __init__(self, weight_path=None) -> None:
        self.weight_path = weight_path
        self.model = None

    def setup_CNN(self, rt=False):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))

        if self.weight_path is not None:
            self.model.load_weights(self.weight_path)
        # model.summary()
        if rt:
            return self.model

    def train(self, data_path: str):
        images, labels = load_data(data_path)
        # build model
        self.setup_CNN(rt=False)
        # compile
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(1e-3), metrics=['acc'])

        early_stopping = EarlyStopping(patience=3)

        # reduce learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1, )

        # Model Checkpoint
        cpt_save = ModelCheckpoint('model/weight.h5', save_best_only=True, monitor='val_acc', mode='max')

        print("Training......")
        self.model.fit(images, labels, callbacks=[cpt_save, reduce_lr, early_stopping], verbose=1, epochs=10, validation_split=0.15, batch_size=32,
                       shuffle=True)
