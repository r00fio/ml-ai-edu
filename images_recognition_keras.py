import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt

# 1. Загружаем MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Добавляем инвертированные картинки
x_train = np.concatenate([x_train, 255 - x_train], axis=0)
y_train = np.concatenate([y_train, y_train], axis=0)

# 3. Преобразуем данные в float и нормализуем
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# 4. One-hot encoding для меток
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 5. Расширяем размерность до (28,28,1) для ImageDataGenerator
x_train_exp = x_train.reshape(-1, 28, 28, 1)

# 6. Аугментация данных
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train_exp)

# 7. Создаем модель Keras
model = Sequential([
    InputLayer(input_shape=(28, 28, 1)),
    Flatten(),               # преобразуем 28x28 в 784
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 8. Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 9. Обучение с аугментацией
model.fit(datagen.flow(x_train_exp, y_train, batch_size=32), epochs=10)

# 10. Тестирование
x_test_exp = x_test.reshape(-1, 28, 28, 1)
loss, acc = model.evaluate(x_test_exp, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# 11. Предсказание для пользовательской картинки
def input_image_from_file(file_path):
    img = Image.open(file_path).convert('L').resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255
    img_array = 1 - img_array  # инвертируем
    return img_array.reshape(1, 28, 28, 1)

def predict_image_file(file_path):
    img_array = input_image_from_file(file_path)
    out = model.predict(img_array)
    pred = np.argmax(out)
    print(f"Predicted digit: {pred}")

    # Визуализация весов первого слоя
    weights, biases = model.layers[2].get_weights()  # Dense 128
    max_neuron = np.argmax(weights.sum(axis=0))
    plt.imshow(weights[:, max_neuron].reshape(28, 28), cmap='gray')
    plt.title(f"Neuron {max_neuron} weights")
    plt.show()

# 12. Цикл ввода
while True:
    file_path = input("Введите путь к изображению (jpg/png/jpeg) или 'exit' для выхода: ")
    if file_path.lower() == 'exit':
        break
    try:
        predict_image_file(file_path)
    except Exception as e:
        print(f"Ошибка: {e}. Попробуйте снова.")
