import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time


def visualize(a): # Plot a digit
	plt.imshow(a[0].numpy(), cmap="Grays")
	plt.xticks([])
	plt.yticks([])
	plt.show()


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

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 120)  # 28x28 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, input):
        # Flatten: output (N, 784)
        l1 = torch.flatten(input, 1)
        # Fully connected: (N, 784) input to (N, 120) output with RELU
        l2 = F.relu(self.fc1(l1))
        # Fully connected: (N, 120) input to (N, 84) output with RELU
        l3 = F.relu(self.fc2(l2))
        # Gaussian layer: (N, 84) input to (N, 10) output
        return self.fc3(l3)


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
batch_size = 100
n_epochs = 10
n_iters = round(n_epochs * len(train_data) / batch_size)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# Create network, optimizer, and loss function
net = Net()
net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()


# Train the model
start_time = time.time()

for it in range(n_iters):
    # Feed in data to current state of network
    data, targets = next(iter(train_loader))
    optimizer.zero_grad()
    out = net(data.to(device))

    # Calculate the loss and backpropagate
    loss = criterion(out, F.one_hot(targets).float().to(device)) # Convert digits to one-hot encoding (2 -> [0,0,1,0,0,0,0,0,0,0])
    loss.backward() # Calculate gradients
    optimizer.step() # New parameters

    if it % 500 == 0:
        print(loss)

end_time = time.time()
print(f"Device: {device}, {n_parameters(net)} model parameters, {n_iters} iterations with batch size {batch_size}, accuracy {accuracy(net, test_loader)}% on out-of-sample data, train time {round(end_time-start_time)} seconds")







