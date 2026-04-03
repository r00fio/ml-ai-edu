class Perceptron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def predict(self, inputs):
        result = 0
        for a in range(len(inputs)):  # размер вектора входов должен совпадать с размером вектора весов
            result += inputs[a] * self.weights[a]
        result += self.bias
        if result > 0:
            return 1
        else:
            return 0


# Первый слой
or_neuron = Perceptron(weights=[1, 1], bias=-0.5)
and_neuron = Perceptron(weights=[1, 1], bias=-1.5)

# Второй слой
final_neuron = Perceptron(weights=[1, -2], bias=-0.5)


def xor(input):
    or_result = or_neuron.predict(input)
    and_result = and_neuron.predict(input)
    final_result = final_neuron.predict([or_result, and_result])
    return final_result


tests = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

for t in tests:
    print(t, "->", xor(t))
