from PIL import Image
import numpy as np
from numpy import dtype, float64, ndarray
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 1. Загружаем данные
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.concatenate([x_train, 255 - x_train], axis=0)
y_train = np.concatenate([y_train, y_train], axis=0)

def augment_image(img):
    # 1. Шум
    noise = np.random.normal(0, 10, img.shape)
    img = img + noise

    # 2. Сдвиг
    shift_x = np.random.randint(-2, 3)
    shift_y = np.random.randint(-2, 3)
    img = np.roll(img, shift_x, axis=0)
    img = np.roll(img, shift_y, axis=1)

    # 3. Обрезаем значения
    img = np.clip(img, 0, 255)

    return img


# применяем ко всему датасету
x_aug = np.array([augment_image(img) for img in x_train])
# объединяем
x_train = np.concatenate([x_train, x_aug])
y_train = np.concatenate([y_train, y_train])

# 2. Превращаем картинки 28x28 в плоский вектор 784
# И нормализуем (делим на 255, чтобы числа были от 0 до 1, а не от 0 до 255)
X = x_train.reshape(240000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255


# 3. Превращаем ответы (0, 1, 2...) в формат для нейросети (One-Hot Encoding)
# Чтобы ответ "3" стал вектором [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
def one_hot(y):
    table = np.zeros((y.size, 10))
    table[np.arange(y.size), y] = 1
    return table


Y = one_hot(y_train)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(y):
    return (y > 0).astype(float)


def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))  # стабилизация
    return exp / np.sum(exp, axis=1, keepdims=True)


# данные XOR
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Y = np.array([[0], [1], [1], [0]])
# скрытый слой 128 нейрона 784 входа 1 выход
W0 = np.random.randn(784, 128) * np.sqrt(2 / 784)
# биасы по одному на нейрон так как w1x1 + w2x2 + b
B0 = np.random.uniform(-1, 1, (1, 128))

W1 = np.random.randn(128, 64) * np.sqrt(2 / 128)
# биасы по одному на нейрон так как w1x1 + w2x2 + b
B1 = np.random.uniform(-1, 1, (1, 64))

# выходной слой 10 нейронов 128 входов 1 выход
W2 = np.random.randn(64, 10) * np.sqrt(2 / 10)
# биасы по одному на нейрон так как w1x1 + w2x2 + b
B2 = np.random.uniform(-1, 1, (1, 10))

lr = 0.01
batch_size = 32


def train(B0, B1, B2, W0, W1, W2, x, y):
    h0 = relu(x @ W0 + B0)
    h1 = relu(h0 @ W1 + B1)
    # выход выходного слоя - матрица Y.size x 10 - 32 теста на 10 нейрон
    out = softmax(h1 @ W2 + B2)

    # ошибка и дельта на выходном слое
    # матрица ошибок по результатам Y.size x 10 (4 теста)
    error_out = out - y
    # вычисляем на сколько быстро растет функция что бы шагнуть вперед
    d_out = error_out * relu_derivative(out)

    error_h1 = d_out @ W2.T
    d_h1 = error_h1 * relu_derivative(h1)

    error_h0 = d_h1 @ W1.T
    d_h0 = error_h0 * relu_derivative(h0)

    # обновляем веса выходного слоя
    W2 -= (h1.T @ d_out) * lr
    # обновляем веса скрытых слоев
    W1 -= (h0.T @ d_h1) * lr

    W0 -= (x.T @ d_h0) * lr

    # суммируем результаты всех 4 тестов у 1 нейрона и предыдущего биаса
    B2 -= np.sum(d_out, axis=0, keepdims=True) * lr
    # суммируем результаты всех 4 тестов у 2 нейронов и предыдущего биаса
    B1 -= np.sum(d_h1, axis=0, keepdims=True) * lr
    # суммируем результаты всех 4 тестов у 2 нейронов и предыдущего биаса
    B0 -= np.sum(d_h0, axis=0, keepdims=True) * lr
    return B0, B1, B2, W0, W1, W2


for epoch in range(20):
    indices = np.random.permutation(len(X))
    X = X[indices]
    Y = Y[indices]
    for j in range(0, len(X), batch_size):
        x_batch = X[j:j + batch_size]
        y_batch = Y[j:j + batch_size]
        # выходы скрытых слоев - матрица batch_size x 128 - 32 теста на 2 нейрона
        B0, B1, B2, W0, W1, W2 = train(B0, B1, B2, W0, W1, W2, x_batch, y_batch)

    print(f"Epoch {epoch + 1} done")

final = softmax(relu(relu(x_test @ W0 + B0) @ W1 + B1) @ W2 + B2)

predicted = np.argmax(final, axis=1)
accuracy = np.mean(predicted == y_test)
print(f"Accuracy: {accuracy * 100}%")


# Ввод пользовательских картинок и вывод нейронов(веса нейронов ввиде картинки) которые предсказали число на картинке


def input_image_from_file(file_path):
    """
    Загружает изображение из файла и преобразует в 28x28 вектор для нейросети.
    Поддерживаются jpg, jpeg, png.
    """
    # Открываем изображение
    img = Image.open(file_path).convert('L')  # преобразуем в grayscale
    img = img.resize((28, 28))  # уменьшаем до 28x28
    img_array = np.array(img, dtype=np.float32)
    img_array = 255 - img_array  # инвертируем цвета (чтобы фон был 0, цифра >0)
    img_array /= 255  # нормализуем на 255
    return img_array


def predict_image_file(file_path):
    """
    Предсказывает число по изображению из файла
    """
    img_array = input_image_from_file(file_path)
    x_flat = img_array.reshape(1, 784)
    out = softmax(relu(relu(x_flat @ W0 + B0) @ W1 + B1) @ W2 + B2)

    pred = np.argmax(out)
    print(f"Predicted digit: {pred}")
    return pred


# Цикл обработки изображений
while True:
    file_path = input("Введите путь к изображению (jpg/png/jpeg) или 'exit' для выхода: ")
    if file_path.lower() == 'exit':
        print("Выход из программы.")
        break
    try:
        predicted_number = predict_image_file(file_path)
    except Exception as e:
        print(f"Ошибка: {e}. Попробуйте снова.")
