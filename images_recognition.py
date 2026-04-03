from PIL import Image
import numpy as np
from numpy import dtype, float64, ndarray
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 1. Загружаем данные
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Превращаем картинки 28x28 в плоский вектор 784
# И нормализуем (делим на 255, чтобы числа были от 0 до 1, а не от 0 до 255)
X = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255


# 3. Превращаем ответы (0, 1, 2...) в формат для нейросети (One-Hot Encoding)
# Чтобы ответ "3" стал вектором [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
def one_hot(y):
    table = np.zeros((y.size, 10))
    table[np.arange(y.size), y] = 1
    return table


Y = one_hot(y_train)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(y):
    return y * (1 - y)


# данные XOR
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Y = np.array([[0], [1], [1], [0]])
# скрытый слой 128 нейрона 784 входа 1 выход
W1 = np.random.uniform(-1, 1, (784, 128))
# биасы по одному на нейрон так как w1x1 + w2x2 + b
B1 = np.random.uniform(-1, 1, (1, 128))

# выходной слой 10 нейронов 128 входов 1 выход
W2 = np.random.uniform(-1, 1, (128, 10))
# биасы по одному на нейрон так как w1x1 + w2x2 + b
B2 = np.random.uniform(-1, 1, (1, 10))

lr = 0.1
batch_size = 32


def train(B1, B2, W1, W2, x, y):
    h = sigmoid(x @ W1 + B1)
    # выход выходного слоя - матрица Y.size x 10 - 32 теста на 10 нейрон
    out = sigmoid(h @ W2 + B2)

    # ошибка и дельта на выходном слое
    # матрица ошибок по результатам Y.size x 10 (4 теста)
    error_out = out - y
    # вычисляем на сколько быстро растет функция что бы шагнуть вперед
    d_out = error_out * sigmoid_derivative(out)

    # вычисляем ошибку скрытого слоя
    # выход: “я ошибся на 0.7”
    # скрытый нейрон:
    # “я влиял на выход через вес 2 → значит моя ошибка ≈ 1.4”
    error_h = d_out @ W2.T  # матрица 4x2 - 4 теста и 2 нейрона
    d_h = error_h * sigmoid_derivative(h)
    # обновляем веса выходного слоя
    W2 -= (h.T @ d_out) * lr
    # суммируем результаты всех 4 тестов у 1 нейрона и предыдущего биаса
    B2 -= np.sum(d_out, axis=0, keepdims=True) * lr
    # обновляем веса скрытых слоев
    W1 -= (x.T @ d_h) * lr
    # суммируем результаты всех 4 тестов у 2 нейронов и предыдущего биаса
    B1 -= np.sum(d_h, axis=0, keepdims=True) * lr
    return B1, B2, W1, W2


for epoch in range(20):
    indices = np.random.permutation(len(X))
    X = X[indices]
    Y = Y[indices]
    for j in range(0, len(X), batch_size):
        x_batch = X[j:j + batch_size]
        y_batch = Y[j:j + batch_size]
        # выходы скрытых слоев - матрица batch_size x 128 - 32 теста на 2 нейрона
        B1, B2, W1, W2 = train(B1, B2, W1, W2, x_batch, y_batch)

    print(f"Epoch {epoch + 1} done")

final = sigmoid(sigmoid(x_test @ W1 + B1) @ W2 + B2)

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
    img = img.resize((28, 28))               # уменьшаем до 28x28
    img_array = np.array(img, dtype=np.float32)
    img_array = 255 - img_array              # инвертируем цвета (чтобы фон был 0, цифра >0)
    img_array /= 255                          # нормализуем на 255
    return img_array

def predict_image_file(file_path):
    """
    Предсказывает число по изображению из файла
    """
    img_array = input_image_from_file(file_path)
    x_flat = img_array.reshape(1, 784)
    out = sigmoid(sigmoid(x_flat @ W1 + B1) @ W2 + B2)
    pred = np.argmax(out)
    print(f"Predicted digit: {pred}")
    return pred

def visualize_activations(img):
    x_flat = img.reshape(1, 784)

    # forward
    h = sigmoid(x_flat @ W1 + B1)
    out = sigmoid(h @ W2 + B2)

    pred = np.argmax(out)
    true = pred

    print(f"Predicted: {pred}, True: {true}")

    # берём значения нейронов скрытого слоя
    activations = h.flatten()

    # сортируем по силе активации
    top_indices = np.argsort(activations)[-16:][::-1]

    fig, axes = plt.subplots(2, 9)

    # показываем картинку
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f"P:{pred} T:{true}")
    axes[0, 0].axis('off')

    # показываем топ-нейроны
    for i, idx in enumerate(top_indices):
        row = (i + 1) // 9
        col = (i + 1) % 9

        weights = W1[:, idx].reshape(28, 28)

        axes[row, col].imshow(weights, cmap='seismic')
        axes[row, col].set_title(f"{idx}: {activations[idx]:.2f}")
        axes[row, col].axis('off')

    plt.show()


# Цикл обработки изображений
while True:
    file_path = input("Введите путь к изображению (jpg/png/jpeg) или 'exit' для выхода: ")
    if file_path.lower() == 'exit':
        print("Выход из программы.")
        break
    try:
        predicted_number = predict_image_file(file_path)
        visualize_activations(input_image_from_file(file_path))
    except Exception as e:
        print(f"Ошибка: {e}. Попробуйте снова.")

