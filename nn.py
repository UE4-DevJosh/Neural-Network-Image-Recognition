import numpy as np
import nnfs
from nnfs.datasets import spiral_data

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist

nnfs.init()

# Dense layer class. This is for layers with fully connected inputs and outputs
class Dense_Layer:

    # Constructor takes the size of inputs and the size of neurons
    def __init__(self, inputs, neurons):
        # Initializing the weights with a matrix of random weights with size (inputs, neurons).
        self.weights = 0.01 * np.random.randn(inputs, neurons)

        # Initializing the biases with a vector of zeroes.
        self.biases = np.zeros((1, neurons))

    # Forward pass for the dense layer
    def forwardPass(self, inputs):
        self.inputs = inputs
        # The forward pass is the dot product of the inputs and weight. Then adding the bias at the end of the dot product.
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backwardPass(self, derivativeValues):
        self.dweights = np.dot(self.inputs.T, derivativeValues)
        self.dbiases = np.sum(derivativeValues, axis=0, keepdims=True)
        self.dinputs = np.dot(derivativeValues, self.weights.T)

# ActivationFunction class. This is the base class for the activation functions.
class ActivationFunction:
    # Constructor. (Is not currently being used)
    def __init__(self) -> None:
        pass

    # Function method. This returns f(inputs) based on the defined activation function
    def f(self, inputs):
        self.output = inputs
    
    # Backward propogation of the activation function
    def backward(self, derivativeValues):
        self.output = 0

# Rectified Linear Units Activation Function. Returns x when x>0 and 0 when x<0
class ReLU(ActivationFunction):
    # Constructor (Again not used. This just calls the super method)
    def __init__(self) -> None:
        super().__init__()
    
    # Function Method. This returns the x when x>0 and 0 when x<0. (Unit Ramp Function)
    def f(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    # Backward propogation of the activation function
    def backward(self, derivativeValues):
        self.dinputs = derivativeValues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Softmax(ActivationFunction):
    # Constructor (Again not used. This just calls the super method)
    def __init__(self) -> None:
        super().__init__()
    
    # Function Method. This returns (e^z)/(sum(e^z))
    def f(self, inputs):
        self.inputs = inputs
        # Since we do not want exploding values, we are subtracting the max value from the inputs from the e^z term.
        numerators = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        denominators = np.sum(numerators, axis=1, keepdims=True)

        self.output = numerators/denominators

    # Backward propogation of the activation function
    def backward(self, derivativeValues):
        self.dinputs = np.empty_like(derivativeValues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, derivativeValues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def __init__(self) -> None:
        pass

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)
    
    def forward(self, y_predicted, y_true):
        pass

    def backward(self, derivativeValues, y_true):
        pass

class CategoricalCrossentropyLoss(Loss):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y_predicted, y_true):
        samples = len(y_predicted)

        # Clip to prevent dividing by 0
        y_predicted_clipped = np.clip(y_predicted, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_predicted_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_predicted_clipped*y_true, axis=1)
        
        negative_ln_likelihoods = -np.log(correct_confidences)
        return negative_ln_likelihoods
    
    def backward(self, derivativeValues, y_true):
        samples = len(derivativeValues)
        labels = len(derivativeValues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = ( -y_true / derivativeValues ) / samples

class SoftmaxLossCategoricalCrossentropyLoss():
    def __init__(self) -> None:
        self.activation = Softmax()
        self.loss = CategoricalCrossentropyLoss()
    
    def f(self, inputs, y_true):
        self.activation.f(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, derivativeValues, y_true):
        samples = len(derivativeValues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = derivativeValues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs/samples

# Stochastic Gradient Descent (SGD)
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates
    
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    
    def post_update_params(self):
        self.iterations += 1

X, y = spiral_data(samples=100, classes=3)

print(y[0])

# print(X.shape)
# print(y.shape)

(X, y), (test_X, test_y) = mnist.load_data()

newX = np.zeros((300,784))
newY = np.zeros(300).astype(int)

for i in range(300):
    print(i)
    newY[i] = int(y[i])
    temporary = np.zeros(784)
    for j in range(len(X[i])):
        for k in range(len(X[i, j])):
            temporary[k + 28*j] = (X[i, j, k]) / 255
    newX[i] = temporary

X = newX
y = newY

print(y[0])

#dense1 = Dense_Layer(2, 64)
dense1 = Dense_Layer(784, 728)
activation1 = ReLU()

#dense2 = Dense_Layer(64, 3)
dense2 = Dense_Layer(728, 10)
loss_activation = SoftmaxLossCategoricalCrossentropyLoss()

#optimizer = Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9)
optimizer = Optimizer_Adam(learning_rate=0.005, decay=5e-7)

for epoch in range(10001):

#Need to split up the batches and use them for the different training. Maybe do 6 batches for 60000

    dense1.forwardPass(X)
    activation1.f(dense1.output)

    dense2.forwardPass(activation1.output)
    loss = loss_activation.f(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, acc: {accuracy}, loss: {loss}, lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backwardPass(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backwardPass(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()