# Libraries
import torch
import torch.nn as nn # for neural network layers
import torch.optim as optim # for optimization algorithms
import torchvision # for image processing
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Load Dataset (CIFAR10 dataset is used)
def get_data_loaders(batch_size=64): # Batch size = number of samples processed in each iteration
  transform = transforms.Compose(
      [
          transforms.ToTensor(), # Convert the image to tensor
          transforms.Normalize(((0.5, 0.5, 0.5)), (0.5, 0.5, 0.5)) # Normalize RGB channels
          ]
  )

  # Download CIFAR10 dataset and create train/test sets
  train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
  test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

  # Dataloader
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

  return train_loader, test_loader

# Visualize dataset
def imshow(img):
  # While downloading the data, we used transform with normalization. 
  # To visualize, we revert normalization back to the original state.
  img = img / 2 + 0.5 # Inverse of normalization
  np_img = img.numpy() # convert tensor to numpy array
  plt.imshow(np.transpose(np_img, (1, 2, 0))) # correctly display colors for 3 channels
  plt.show()

def get_sample_images(train_loader): # Function to get sample images from dataset
  data_iter = iter(train_loader)
  images, labels = next(data_iter)
  return images, labels

def visualize(n):

  train_loader, test_loader = get_data_loaders()
  # visualize n samples
  images, labels = get_sample_images(train_loader)
  plt.figure()
  for i in range(n):
    plt.subplot(1, n, i + 1)
    imshow(images[i]) # visualization
    plt.title(f"label: {labels[i].item()}")
    plt.axis("off")
  plt.show()

visualize(10)

# Build CNN Model

class CNN(nn.Module):

  def __init__(self):
    super(CNN, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) # in_channels=3(rgb channels), out_channels=32(number of filters), kernel_size=3x3
    self.relu = nn.ReLU() # Activation function
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Pooling layer of size 2x2
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # second convolution layer with 64 filters
    self.dropout = nn.Dropout(0.2) # dropout works at 20%
    # image(3x32x32) -> conv(32x32x32) -> relu(32x32x32) -> pool(32x16x16)
    # conv(64x16x16) -> relu(64x16x16) -> pool(64x8x8) -> image(64x8x8)
    self.fc1 = nn.Linear(64*8*8, 128) # 64 filters * 8x8 image shape = 4096, fully connected input=4096, output=128
    self.fc2 = nn.Linear(128, 10) # Output layer

  def forward(self, x):
    """
      image(3x32x32) -> conv(32x32x32) -> relu(32x32x32) -> pool(32x16x16)
      conv(64x16x16) -> relu(64x16x16) -> pool(64x8x8) -> image(64x8x8)
      flatten
      fc1 -> relu -> dropout
      fc2 -> output
    """
    x = self.pool(self.relu(self.conv1(x))) # First Convolution Block
    x = self.pool(self.relu(self.conv2(x))) # Second Convolution Block
    x = x.view(-1, 64*8*8) # Flatten
    x = self.dropout(self.relu(self.fc1(x))) # Fully Connected Layer
    x = self.fc2(x) # output

    return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device

model = CNN().to(device)

# Define loss funtion and optimizer
define_loss_and_optimizer = lambda model:(
    nn.CrossEntropyLoss(),
    optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # momentum -> a method that accelerates SGD (Stochastic Gradient Descent) and prevents getting stuck in local minima
)

# Train

def train_model(model, train_loader, criterion, optimizer, epochs=5):

  model.train() # set model to training mode

  train_losses = [] # list to store loss values

  for epoch in range(epochs): # For loop for given number of epochs
    total_loss = 0 # to store total loss
    for images, labels in train_loader: # iterate over training dataset
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad() # reset gradients
      outputs = model(images) # forward propagation (prediction)
      loss = criterion(outputs, labels) # compute loss value
      loss.backward() # Back-propagation (compute gradients)
      optimizer.step() # Learning = update weights (parameters)

      total_loss += loss.item()

    avg_loss = total_loss / len(train_loader) # compute average loss
    train_losses.append(avg_loss)
    print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")

  # Loss Graph
  plt.figure()
  plt.plot(range(1, epochs+1), train_losses, marker="o", linestyle="-", label="Train Loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.title("Training Loss")
  plt.legend()
  plt.show()

train_loader, test_loader = get_data_loaders()
model = CNN().to(device)
criterion, optimizer = define_loss_and_optimizer(model)
train_model(model, train_loader, criterion, optimizer, epochs=10)

# Test

def test_model(model, test_loader, dataset_type):
  model.eval() # Evaluation mode
  correct = 0 # Correct predictions counter
  total = 0 # Total data counter

  with torch.no_grad(): # Disable gradient computation
    for images, labels in test_loader: # Evaluate using test dataset
      images, labels = images.to(device), labels.to(device) # move data to device

      outputs = model(images) # prediction
      _, predicted = torch.max(outputs, 1) # choose the class with highest probability
      total += labels.size(0) # total number of samples
      correct += (predicted == labels).sum().item() # count correct predictions

  print(f"{dataset_type} accuracy: {100 * correct / total}%") # print accuracy

test_model(model, test_loader, dataset_type="test")

test_model(model, train_loader, dataset_type="train")

# Main Program
if __name__ == "__main__":
  # Load dataset
  train_loader, test_loader = get_data_loaders()

  # Visualization
  visualize(10)

  # Training
  model = CNN().to(device)
  criterion, optimizer = define_loss_and_optimizer(model)
  train_model(model, train_loader, criterion, optimizer, epochs=10)

  # Test
  test_model(model, test_loader, dataset_type="test")
  test_model(model, train_loader, dataset_type="train")
