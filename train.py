#!/usr/bin/env python
# Check whether CUDA is installed and active
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer

if torch.cuda.is_available():
    print("CUDA is available, now using device " + str(torch.cuda.current_device())+'.')
else:
    print("CUDA not detected, running on CPU.")

from dotenv import load_dotenv
import os
import time
import sys

load_dotenv('.env')
training_folder = os.getenv("TRAINING")
nb_items = len(os.listdir(training_folder))
print('Reading from: ' + training_folder)
print('Found ' + str(nb_items) + ' items in the training directory.')
if nb_items % 2 != 0:
    print('An odd number of items were detected. Make sure you have 1 tag file for 1 image!')
    if sys.platform == 'darwin':
        print('This error may be caused by the presence of a .DS_Store file in your folder. Check with "ls -a | grep .DS_Store" to confirm.\n')

model_name = os.getenv("MODEL")
if os.path.exists(model_name):
    print('A model with the provided name was found. Training will overwrite it!')
    print('Press CTRL+C to cancel the script. Continuing in:')
    for i in range(1,6):
        print(6-i)
        time.sleep(1)
else:
    print('No model found. It will be created after training.')

tags_pickle = os.getenv("TAGS")
print('Beginning training...')

# Training
torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 10
batch_size = 256

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, mlb=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mlb = mlb
        self.image_paths, self.tags = self.load_data()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):   # Added this method
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Convert image to RGB
        image = self.transform(image)
        labels = torch.FloatTensor(self.tags[idx])  # Convert labels to tensor
        return image, labels

    def load_data(self):
        images = []
        tags = []
        for filename in os.listdir(self.root_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                images.append(os.path.join(self.root_dir, filename))
                tag_file = filename + ".txt"
                tag_path = os.path.join(self.root_dir, tag_file)
                with open(tag_path, 'r') as f:
                    tag_list = f.read().splitlines()
                tags.append(set(tag_list))
        return images, tags


shuffle = True
image_folder = training_folder

all_tags = []
data_tags = []  # Initialize a list to store the sets of tags for each image
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        tag_file = filename + ".txt"
        tag_path = os.path.join(image_folder, tag_file)
        with open(tag_path, 'r') as f:
            tag_list = f.read().splitlines()
        all_tags.extend(tag_list)
        data_tags.append(set(tag_list))  # Store the set of tags for each image

tags_set = set(all_tags)

mlb = MultiLabelBinarizer(classes=sorted(tags_set))
mlb.fit([tags_set])  # fit on the total tags_set

data_tags_binary = mlb.transform(data_tags)  # Transform tag sets to binary matrix

dataset = CustomDataset(image_folder, transform=transform, mlb=mlb)

dataset.tags = data_tags_binary.tolist()


train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)


# Load the pre-trained ResNet model
resnet = models.resnet101(weights=True)

# Replace the last fully connected layer to match the number of classes in your dataset
resnet.fc = nn.Linear(resnet.fc.in_features, len(mlb.classes_))

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss() # Binary Cross-Entropy Loss Function
optimizer = torch.optim.Adam(resnet.fc.parameters()) # Adam optimizer (it works effectively for deep learning models)

# Initialize the metric variables
best_val_loss = float('inf')

# Initialization before training of the confusion matrix
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

# The training loop
resnet.to(device)
resnet.train()
for epoch in range(num_epochs):
    running_loss = 0.0 # This reset to zero for every new epoch
    # Training Phase
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() # Resetting gradients to zero (prevents unwanted gradient accumulation)

        logits = resnet(inputs) # Forward pass
        loss = criterion(logits, labels) # Loss calculation

        loss.backward() # Backward pass (gradient calculation)
        optimizer.step() # Weight update

        running_loss += loss.item() * inputs.size(0) # Multiplying with batch size to get the loss for the whole batch

    epoch_loss = running_loss / len(train_loader.dataset) # Average loss in one epoch

    # Validation Phase
    running_loss = 0.0
    resnet.eval() # Switch model to evaluation mode
    with torch.no_grad(): # Turn off gradients for validation, saves memory and computations
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = resnet(inputs) # Forward pass
            loss = criterion(logits, labels) # Loss calculation

            running_loss += loss.item() * inputs.size(0) # Multiply with batch size

            # Calculate performance metrics
            predictions = torch.round(torch.sigmoid(logits)) # Adding sigmoid to convert logits to (0,1) & rounding to get prediction
            true_positive += ((predictions == 1) & (labels == 1)).sum().item()
            false_positive += ((predictions == 1) & (labels == 0)).sum().item()
            false_negative += ((predictions == 0) & (labels == 1)).sum().item()
            true_negative += ((predictions == 0) & (labels == 0)).sum().item()

    val_loss = running_loss / len(val_loader.dataset) # Average loss in one epoch

    print(f'Epoch {epoch+1}/{num_epochs}:')
    print('Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch+1, epoch_loss, val_loss))
    print(f'TP={true_positive} \tFP={false_positive} \tFN={false_negative} \tTN={true_negative}')

    # Save the model with the best-validation-loss weights (overwrites previous best model)
    if val_loss < best_val_loss:
        print('Validation Loss Decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_val_loss,val_loss))
        torch.save(resnet.state_dict(), 'model.pt')
        best_val_loss = val_loss


# Loading the best model (with least validation loss)

model = models.resnet101(weights=True)
model.fc = nn.Linear(model.fc.in_features, len(mlb.classes_))
model.load_state_dict(torch.load('model.pt'))

# Saving
import pickle
# Path to save model
model_path = model_name
# Save the trained model
torch.save(resnet.state_dict(), model_path)
# Save the unique set of classes
with open(tags_pickle, 'wb') as f:
    pickle.dump(list(mlb.classes_), f)

# Print list of tags
print(mlb.classes_)