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

# Load Data From MNIST Dataset
(X, y), (test_X, test_y) = mnist.load_data()

baseDimX, baseDimY = 60000, 784

# Batch Size of Training. Total Epoch Number will be total length divided by batchSize
batchSize = 128
totalEpochNumber = int(baseDimX / batchSize)

newX = np.zeros((totalEpochNumber, batchSize, 784))
newY = np.zeros((totalEpochNumber, batchSize))

# # Looping through each MNIST training and test data
# for i in range(baseDimX):

#     epochNumber = int(i / batchSize)
#     batchNumber = int(i % batchSize)

#     if (epochNumber) >= totalEpochNumber:
#         break

#     print(f"Epoch {epochNumber}, Batch {batchNumber}")

#     # Array to hold one number
#     temporary = np.zeros(784)

#     for j in range(len(X[i])):
#         for k in range(len(X[i, j])):
#             # Normalize by dividing to 255
#             temporary[k + 28*j] = (X[i, j, k]) / 255

#     newX[epochNumber, batchNumber] = temporary
#     newY[epochNumber, batchNumber] = y[i]

# X = newX
# # Must be an integer since the array will be one-hot values
# y = newY.astype(int)

# np.save('x.npy', X)
# np.save('y.npy', y)

X = np.load('x.npy')
y = np.load('y.npy')

dense1 = Dense_Layer(784, 512)
activation1 = ReLU()

dense2 = Dense_Layer(512, 256)
activation2 = ReLU()

dense3 = Dense_Layer(256, 128)
activation3 = ReLU()

dense4 = Dense_Layer(128, 10)
loss_activation = SoftmaxLossCategoricalCrossentropyLoss()

#optimizer = Optimizer_SGD(learning_rate=1e-4, decay=1e-3, momentum=0.9)
optimizer = Optimizer_Adam(learning_rate=0.0005, decay=5e-7)

print(X[1], y[1])

for epoch in range(totalEpochNumber):

    currentX, currentY = X[epoch], y[epoch]

    dense1.forwardPass(currentX)
    activation1.f(dense1.output)

    dense2.forwardPass(activation1.output)
    activation2.f(dense2.output)

    dense3.forwardPass(activation2.output)
    activation3.f(dense3.output)

    dense4.forwardPass(activation3.output)
    loss = loss_activation.f(dense4.output, currentY)

    predictions = np.argmax(loss_activation.output, axis=1)
    
    accuracy = np.mean(predictions == currentY)

    loss_activation.backward(loss_activation.output, currentY)
    dense4.backwardPass(loss_activation.dinputs)

    activation3.backward(dense4.dinputs)
    dense3.backwardPass(activation3.dinputs)

    activation2.backward(dense3.dinputs)
    dense2.backwardPass(activation2.dinputs)

    activation1.backward(dense2.dinputs)
    dense1.backwardPass(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.update_params(dense4)
    optimizer.post_update_params()

    if not epoch % 10:
        print(f'epoch: {epoch}, acc: {accuracy}, loss: {loss}, lr: {optimizer.current_learning_rate}')

