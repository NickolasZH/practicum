from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

# Функция загрузки обучающей выборки
def load_train():
    # Загружаем датасет Fashion MNIST
    (features_train, target_train), (features_test, target_test) = fashion_mnist.load_data()
    
    # Приводим яркость изображений к диапазону [0,1]
    features_train = features_train.reshape(features_train.shape[0], 28 * 28) / 255.0
    features_test = features_test.reshape(features_test.shape[0], 28 * 28) / 255.0
    
    return (features_train, target_train), (features_test, target_test)

# def load_train(path):
#     features_train = np.load(path + 'train_features.npy')
#     target_train = np.load(path + 'train_target.npy')
#     features_train = features_train.reshape(features_train.shape[0], 28, 28, 1) / 255.0
    
#     return features_train, target_train

# Функция создания модели
def create_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),  # Преобразование 2D в 1D
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # Выходной слой для классификации (10 классов)
    ])
    
    # Компиляция модели с Adam-оптимизатором
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Функция обучения модели
def train_model(model, train_data, test_data, batch_size=32, epochs=10, 
                steps_per_epoch=None, validation_steps=None):
    
    features_train, target_train = train_data
    features_test, target_test = test_data
    
    # Обучение модели
    model.fit(features_train, target_train, 
              validation_data=(features_test, target_test),
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)
    
    return model

# Загружаем данные
train_data, test_data = load_train()

# Создаём модель
model = create_model((28 * 28,))

# Обучаем модель
trained_model = train_model(model, train_data, test_data, epochs=10)

# Оцениваем качество модели на тестовой выборке
test_loss, test_accuracy = trained_model.evaluate(test_data[0], test_data[1], verbose=2)

print(f"Точность на тестовой выборке: {test_accuracy * 100:.2f}%")
