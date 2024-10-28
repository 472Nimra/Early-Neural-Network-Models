# Neural Network Implementations: Perceptron, MLP, and Hopfield Network

## Project Overview

This project showcases the fundamental workings of three neural network architectures:
1. **Perceptron**: A simple linear classifier capable of performing binary classification.
2. **Multi-Layer Perceptron (MLP)**: A feedforward neural network that can model complex functions, including non-linear relationships.
3. **Hopfield Network**: A recurrent neural network that serves as an associative memory, capable of pattern retrieval even in the presence of noise.

## Key Concepts

### 1. Perceptron

The Perceptron is the simplest form of a neural network, introduced by Frank Rosenblatt in 1958. It consists of a single layer of output nodes connected to input features. The Perceptron updates its weights based on the input data and the associated labels during training.

#### Key Components:
- **Weights**: Parameters that are adjusted during training to minimize prediction error.
- **Activation Function**: A step function that determines the output based on the weighted sum of inputs.

### 2. Multi-Layer Perceptron (MLP)

The MLP consists of one or more hidden layers between the input and output layers, enabling it to learn non-linear mappings from inputs to outputs. The training process uses **backpropagation**, which involves two main steps:
- **Feedforward**: Compute the output of the network.
- **Backpropagation**: Adjust weights based on the error between the predicted and actual outputs.

#### Key Components:
- **Hidden Layers**: Intermediate layers that help capture complex patterns.
- **Activation Functions**: Non-linear functions like the sigmoid function, used to introduce non-linearity into the model.

### 3. Hopfield Network

The Hopfield Network is a recurrent neural network that can store and recall patterns. It is characterized by its symmetrical weight matrix and the ability to converge to a stored pattern even from a noisy input.

#### Key Components:
- **Hebbian Learning**: A learning rule used to store patterns in the weight matrix.
- **Synchronous Update**: The state of each neuron is updated simultaneously based on the current state of all neurons.

## Code Explanation

### 1. Perceptron Implementation

```python
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.zeros(input_size + 1)  # Including bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        summation = np.dot(x, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def train(self, X, y):
        for _ in range(self.epochs):
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
```

- **Initialization**: The constructor initializes weights, learning rate, and number of training epochs.
- **Activation Function**: A simple step function determines output based on a weighted sum.
- **Predict Method**: Computes the prediction for a given input.
- **Train Method**: Iteratively updates the weights based on the predictions and actual labels.

#### Training the Perceptron

```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND operation
perceptron = Perceptron(input_size=2)
perceptron.train(X, y)

print("Perceptron Predictions (AND Gate):")
for inputs in X:
    print(f"Input: {inputs}, Prediction: {perceptron.predict(inputs)}")
```

- The Perceptron is trained on an AND operation dataset. After training, it prints predictions for each input.

### 2. Multi-Layer Perceptron (MLP) Implementation

```python
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        # Forward pass
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backpropagate(self, X, y, output):
        # Backward pass
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * self.learning_rate

    def train(self, X, y):
        for _ in range(self.epochs):
            output = self.feedforward(X)
            self.backpropagate(X, y, output)
```

- **Initialization**: The constructor initializes weights and biases for input, hidden, and output layers.
- **Activation Functions**: The sigmoid function is used for both the hidden and output layers, with its derivative calculated for backpropagation.
- **Feedforward**: Computes the output of the network through the hidden and output layers.
- **Backpropagation**: Updates the weights and biases based on the output error.

#### Training the MLP

```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR operation
mlp = MLP(input_size=2, hidden_size=4, output_size=1)
mlp.train(X, y)

print("\nMLP Predictions (XOR Gate):")
for inputs in X:
    print(f"Input: {inputs}, Prediction: {mlp.feedforward(inputs)[0]:.3f}")
```

- The MLP is trained on an XOR operation dataset. After training, it prints predictions for each input, demonstrating its ability to learn non-linear functions.

### 3. Hopfield Network Implementation

```python
class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        # Train using Hebbian learning
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)

    def predict(self, pattern, steps=10):
        # Update neurons synchronously
        output = pattern.copy()
        for _ in range(steps):
            output = np.sign(self.weights.dot(output))
        return output
```

- **Initialization**: The constructor initializes the weight matrix for the Hopfield network.
- **Training**: The `train` method uses Hebbian learning to create the weight matrix by computing the outer product of each pattern. The diagonal is set to zero to prevent self-connections.
- **Prediction**: The `predict` method updates the neuron states synchronously for a specified number of steps based on the current state.

#### Training the Hopfield Network

```python
patterns = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]])  # Memorize two patterns
hopfield_net = HopfieldNetwork(num_neurons=4)
hopfield_net.train(patterns)

# Testing Hopfield Network with noisy input
test_pattern = np.array([1, -1, -1, -1])  # Slightly noisy pattern
output_pattern = hopfield_net.predict(test_pattern)

print("\nHopfield Network Prediction (Pattern Recovery):")
print(f"Test Pattern: {test_pattern}")
print(f"Recovered Pattern: {output_pattern}")
```

- The Hopfield Network is trained with two patterns. It then tests the network's ability to recover a stored pattern from a noisy input.

## Conclusion

This project illustrates the fundamental principles of neural networks, showing how they can be implemented from scratch using Python and NumPy. Each network serves a different purpose, from simple classification (Perceptron) to more complex function approximations (MLP) and pattern retrieval (Hopfield Network). The example demonstrates the capabilities and differences among these neural network architectures.
