# Library
import torch # PyTorch library for tensor operations
import torch.nn as nn # Used to define artificial neural network layers
import torch.optim as optim # Module containing optimization algorithms
import torchvision # Includes image processing and pre-defined models
import torchvision.transforms as transforms # For performing image transformations
import matplotlib.pyplot as plt # Visualization

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device

# Veri seti yükleme, Data Loading
def get_data_loaders(batch_size=64): # Amount of data to be processed in each iteration, batch size
  transform = transforms.Compose([
      transforms.ToTensor(), # Convert image to tensor and scale from 0-255 to 0-1
      transforms.Normalize((0.5,), (0.5,))
      ])

  # Download MNIST dataset and create train/test sets
  train_set = torchvision.datasets.MNIST(root= "./data", train = True, download=True, transform = transform) # ./ means this folder. If 'data' folder exists use it, else create it.
  test_set = torchvision.datasets.MNIST(root= "./data", train = False, download=True, transform = transform)

  # Create PyTorch data loader
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

  return train_loader, test_loader

train_loader, test_loader = get_data_loaders()

# Data Visualization
def visualize_samples(loader, n):
  images, labels = next(iter(loader)) # Get images and labels from the first batch
  fig, axes = plt.subplots(1, n, figsize=(10, 5)) # Visualization area for n different images
  for i in range(n):
    axes[i].imshow(images[i].squeeze(), cmap="gray") # Display the image in grayscale
    axes[i].set_title(f"Label: {labels[i].item()}") # Write class labels of the image as title
    axes[i].axis("off") # Hide axes
  plt.show()

visualize_samples(train_loader, 4)

# Define ANN Model

## Artificial Neural Network Class

class NeuralNetwork(nn.Module): # Inherits from PyTorch's nn.Module class
  def __init__(self): # Define the components necessary to build the neural network
    super(NeuralNetwork, self).__init__()

    # Flatten 2D images into 1D vectors
    self.flatten = nn.Flatten()

    # Create the first fully connected layer
    self.fc1 = nn.Linear(28*28, 128) # Fist Fully Connected layer : 784 = input size, 128 = output size

    # Create activation function
    self.relu = nn.ReLU()

    # Create the second fully connected layer
    self.fc2 = nn.Linear(128, 64) # Second Fully Connected layer : 128 = input size, 64 = output size

    # Create output layer
    self.fc3 = nn.Linear(64, 10) # input  size = 64, output size = 10

  def forward(self, x): # Forward propagation, input x = image

    x = self.flatten(x) # initial x = 28 * 28 of images -> Flatten to 784 vectors
    x = self.fc1(x) # First fully connected layer
    x = self.relu(x) # Activation function
    x = self.fc2(x) # Second fully connected layer
    x = self.relu(x) # Activation function
    x = self.fc3(x) # Output layer

    return x

model = NeuralNetwork().to(device)

# Determining the loss function and optimization algorithm

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), # Loss function for multi-class classification problems
    optim.Adam(model.parameters(), lr=0.001) # Update weights with Adam optimizer
)

criterion, optimizer = define_loss_and_optimizer(model)

# Train

def train_model(model, train_loader, criterion, optimizer, epochs = 10):

  # Set model to training mode
  model.train()

  # List to store loss values for each epoch
  train_losses = []

  # Train for the specified number of epochs
  for epoch in range(epochs):
    total_loss = 0 # Total loss value

    for images, labels in train_loader: # Iterate over all training data
      images, labels = images.to(device), labels.to(device) # Move data to GPU

      optimizer.zero_grad() # Reset gradients

      predictions = model(images) # Apply the model, forward propagation

      loss = criterion(predictions, labels) # Compute loss -> y_prediction vs y_real

      loss.backward() # Backward propagation, gradient computation

      optimizer.step() # Update weights

      total_loss = total_loss + loss.item()

    avg_loss = total_loss / len(train_loader) # Compute average loss
    train_losses.append(avg_loss)
    print(f"Epoch: {epoch + 1}/{epochs}, Loss: {avg_loss:.3f} ")

  # Loss graph
  plt.figure()
  plt.plot(range(1, epochs + 1), train_losses, marker="o", linestyle="-", label="Train Loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.title("Training Loss")
  plt.legend()
  plt.show()

train_model(model, train_loader, criterion, optimizer, epochs=5)

# Test

def test_model(model, test_loader):
  model.eval() # Set model to evaluation mode
  correct = 0 # Correct predictions counter
  total = 0 # Total samples counter

  with torch.no_grad(): # Disabled gradient computation as it is unnecessary
    for images, labels in test_loader: # Move data to device
      images, labels = images.to(device), labels.to(device)
      predictions = model(images)
      _, predicted = torch.max(predictions, 1) # Gives the index of the maximum (class with highest probability)
      total += labels.size(0) # Update total sample count
      correct += (predicted == labels).sum().item() # Count correct predictions

  print(f"Test Accuracy: {100 * correct/total:.3f}%")

test_model(model, test_loader)

# Main

if __name__ == "__main__":
  train_loader, test_loader = get_data_loaders() # Get data loaders
  visualize_samples(train_loader, 5)
  model = NeuralNetwork().to(device)
  criterion, optimizer = define_loss_and_optimizer(model)
  train_model(model, train_loader, criterion, optimizer)
  test_model(model, test_loader)