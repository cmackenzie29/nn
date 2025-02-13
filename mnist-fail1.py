import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time


### Failed because BTB ends up being singular for pretty much any model architecture


def n_parameters(net): # Return the total number of parameters in the network
    n = 0
    for l in list(net.parameters()):
        n += l.numel() # Number of elements in each tensor
    return n


def accuracy(net, test_loader): # Return the percentage of accurate predictions by the network
    correct = 0
    total = 0
    for data, targets in test_loader:
        out = net(data.to(device))
        _, predictions = torch.max(out.data, 1) # Item with highest value in one hot is the prediction
        correct += int(sum(predictions == targets.to(device)))
        total += len(predictions)
    
    return 100*correct/total


class Net(torch.nn.Module):

    def __init__(self, inner_layers=None): # number of neurons in inner hidden layers passed in as list of ints
        super().__init__()

        self.layers = torch.nn.Sequential()
        
        if (inner_layers == None) or len(inner_layers) == 0:
            self.layers.append(torch.nn.Linear(28*28, 10))
        else:
            self.layers.append(torch.nn.Linear(28*28, inner_layers[0])) # First layer is 28x28 -> l[0]
            
            for l in range(len(inner_layers)-1): # Specify all inner layers
                self.layers.append(torch.nn.Linear(inner_layers[l], inner_layers[l+1]))

            self.layers.append(torch.nn.Linear(inner_layers[-1], 10)) # Last layer is l[-1] -> 10 outputs

    def forward(self, input):
        # Flatten: output (N, 784)
        out = torch.flatten(input, 1)
        
        # First through second to last fully connected layers with RELU
        for l in range(len(self.layers)-1):
            out = F.gelu(self.layers[l](out))

        # Final layer is Gaussian to (N, 10) output
        return self.layers[-1](out)

    # Change number of neurons while approximately preserving existing relationships
    # See documentation for method details
    # n_neurons can be a positive or negative integer to change the number of neurons
    # layer is zero-indexed and cannot be 0 (input dimension is fixed) or len(layers) (output dimension is fixed)
    def modify_neurons(self, n_neurons, layer, device):
        if (type(n_neurons) != int) or (type(layer) != int):
            raise Exception("Parameters must be integers")
        elif layer > len(self.layers):
            raise Exception(f"There are not {layer+1} layers in the network")
        elif (layer <= 0) or (layer == len(self.layers)):
            raise Exception("Input and output dimensions of the network are fixed")

        inner_layers = [x.out_features for x in self.layers][:-1]
        inner_layers[layer-1] += n_neurons

        if inner_layers[layer-1] <= 0:
            raise Exception("Too many neurons removed")

        new_net = Net(inner_layers)
        new_net.to(device)
    
        BTBinv = torch.inverse(new_net.layers[layer].weight.T @ new_net.layers[layer].weight)
        with torch.no_grad():
            # A* = (B*T B*)-1 B*T B A
            # a* = (B*T B*)-1 B*T (Ba + b - b*)
            new_net.layers[layer-1].weight.data = BTBinv @ new_net.layers[layer].weight.T @ self.layers[layer].weight @ self.layers[layer-1].weight
            new_net.layers[layer-1].bias.data = BTBinv @ new_net.layers[layer].weight.T @ (self.layers[layer].weight @ self.layers[layer-1].bias + self.layers[layer].bias - new_net.layers[layer].bias)

        return new_net


    # Add new layer while approximately preserving existing relationships
    # See documentation for method details
    # n_neurons must be a positive integer indicating the number of neurons in the new layer
    # after_layer is zero-indexed and cannot be len(layers) (i.e. after the end of the network)
    def add_layer(self, n_neurons, after_layer, device):
        if (type(n_neurons) != int) or (type(after_layer) != int):
            raise Exception("Parameters must be integers")
        elif after_layer >= len(self.layers):
            raise Exception(f"{after_layer} is after the end of the network")
        elif n_neurons <= 0:
            raise Exception("Must specify a positive number of neurons to add")
        elif after_layer < 0:
            raise Exception("Must specify a non-negative layer")

        inner_layers = [x.out_features for x in self.layers][:-1]
        inner_layers.insert(after_layer, n_neurons)
        
        new_net = Net(inner_layers)
        new_net.to(device)
    
        BTBinv = torch.inverse(new_net.layers[after_layer+1].weight.T @ new_net.layers[after_layer+1].weight)
        with torch.no_grad():
            # A* = (B*T B*)-1 B*T A
            # a* = (B*T B*)-1 B*T (a - b*)
            new_net.layers[after_layer].weight.data = BTBinv @ new_net.layers[after_layer+1].weight.T @ self.layers[after_layer].weight
            new_net.layers[after_layer].bias.data = BTBinv @ new_net.layers[after_layer+1].weight.T @ (self.layers[after_layer].bias - new_net.layers[after_layer+1].bias)

        return new_net


    # Remove layer while approximately preserving existing relationships
    # See documentation for method details
    # layer is zero-indexed and cannot be 0 or len(layers) (first & last layer dimensions are fixed)
    def drop_layer(self, layer, device):
        if type(layer) != int:
            raise Exception("Parameter must be an integer")
        elif (layer >= len(self.layers)) or (layer <= 0):
            raise Exception(f"Specify an inner layer to drop")

        inner_layers = [x.out_features for x in self.layers][:-1]
        inner_layers.pop(layer-1)
        
        new_net = Net(inner_layers)
        new_net.to(device)
    
        with torch.no_grad():
            # A* = B A
            # a* = B a + b
            new_net.layers[layer-1].weight.data = self.layers[layer].weight @ self.layers[layer-1].weight
            new_net.layers[layer-1].bias.data = self.layers[layer].weight @ self.layers[layer-1].bias + self.layers[layer].bias

        return new_net


    
                


# Set seed for reproducibility
torch.manual_seed(0)


# Initialize device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available(): # M1 chip
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Load dataset
train_data = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_data = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
batch_size = 60
n_epochs = 1
n_iters = round(n_epochs * len(train_data) / batch_size)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# Create network, optimizer, and loss function
net = Net(inner_layers=[])
net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()


# Train the model
start_time = time.time()

for _ in range(n_iters):
    # Feed in data to current state of network
    data, targets = next(iter(train_loader))
    optimizer.zero_grad()
    out = net(data.to(device))

    # Calculate the loss and backpropagate
    loss = criterion(out, F.one_hot(targets, 10).float().to(device)) # Convert digits to one-hot encoding (2 -> [0,0,1,0,0,0,0,0,0,0])
    loss.backward() # Calculate gradients
    optimizer.step() # New parameters


end_time = time.time()
print(f"Device: {device}, {n_parameters(net)} model parameters, {n_iters} iterations with batch size {batch_size}, accuracy {accuracy(net, test_loader)}% on out-of-sample data, train time {round(end_time-start_time)} seconds")







