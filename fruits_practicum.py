from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np


def load_train(path):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1/255.
    )

    train_datagen_flow = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='sparse',
        seed=12345
    )

    return train_datagen_flow


def create_model(input_shape):
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),

        Conv2D(256, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),

        Conv2D(512, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),

        GlobalAveragePooling2D(),  # Уменьшаем количество параметров
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(12, activation='softmax')  # Количество классов
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Уменьшаем скорость обучения
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model




def train_model(model, train_data, test_data, batch_size=None, epochs=5,
                steps_per_epoch=None, validation_steps=None):

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)

    return model