import torch

torch.manual_seed(2023)


def activation_func(x):
    #TODO Implement one of these following activation function: sigmoid, tanh, ReLU, leaky ReLU
    sigmoid = lambda x: 1 / (1 + torch.exp(-x))
    #epsilon = 0.01   # Only use this variable if you choose Leaky ReLU
    result = sigmoid(x)
    return result

def softmax(x):
    # TODO Implement softmax function here
    exp_x = torch.exp(x - torch.max(x))  # Subtract max for numerical stability
    result = exp_x / torch.sum(exp_x, dim=1, keepdim=True)  # Normalize to get probabilities
    return result

result = activation_func(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))  # Test activation function
print(result)
result_softmax = softmax(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))  # Test softmax function
print(result_softmax)
# Define the size of each layer in the network
num_input = 784  # Number of node in input layer (28x28)
num_hidden_1 = 128  # Number of nodes in hidden layer 1
num_hidden_2 = 256  # Number of nodes in hidden layer 2
num_hidden_3 = 128  # Number of nodes in hidden layer 3
num_classes = 10  # Number of nodes in output layer

# Random input
input_data = torch.randn((1, num_input))
# Weights for inputs to hidden layer 1
W1 = torch.randn(num_input, num_hidden_1)
# Weights for hidden layer 1 to hidden layer 2
W2 = torch.randn(num_hidden_1, num_hidden_2)
# Weights for hidden layer 2 to hidden layer 3
W3 = torch.randn(num_hidden_2, num_hidden_3)
# Weights for hidden layer 3 to output layer
W4 = torch.randn(num_hidden_3, num_classes)

# and bias terms for hidden and output layers
B1 = torch.randn((1, num_hidden_1))
B2 = torch.randn((1, num_hidden_2))
B3 = torch.randn((1, num_hidden_3))
B4 = torch.randn((1, num_classes))

#TODO Calculate forward pass of the network here. Result should have the shape of [1,10]
# Dont forget to check if sum of result = 1.0
result = input_data @ W1 + B1  # Input to hidden layer 1
result = activation_func(result)  # Activation function for hidden layer 1
result = result @ W2 + B2  # Hidden layer 1 to hidden layer 2
result = activation_func(result)  # Activation function for hidden layer 2
result = result @ W3 + B3  # Hidden layer 2 to hidden layer 3
result = activation_func(result)  # Activation function for hidden layer 3
result = result @ W4 + B4  # Hidden layer 3 to output layer
result = softmax(result)  # Apply softmax to get probabilities
# Check if the sum of the result is 1.0
assert torch.isclose(result.sum(), torch.tensor(1.0)), "The sum of the result is not equal to 1.0"
# Print the final result
print(result)
