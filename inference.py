import pickle
from PIL import Image
import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torchvision import transforms, models

# Preloads and checks
load_dotenv()
model_path = os.getenv("MODEL")
image_dir = os.getenv("INFERENCE_DIR")
tags = os.getenv("TAGS")
whitelist_file = 'whitelist.txt'
print('Loading model from: ' + model_path+' and tags from: '+tags)
if os.path.exists(model_path) and os.path.exists(tags):
    print('Model and tags found. Proceeding...')
elif os.path.exists(model_path) and not os.path.exists(tags):
    print('Tags not found.')
    exit()
elif not os.path.exists(model_path) and os.path.exists(tags):
    print('Model not found.')
    exit()
else:
    print('Neither model nor tags found. Check your directory path!')
    exit()
if os.path.exists(whitelist_file):
    print('Whitelist found. Only selected tags will be used.')
else:
    print('Whitelist not found. All tags will be used.')
print('Tagging images from: ' + image_dir)


# Loading and Inference
# Load pickle file for tags_set
with open(tags, 'rb') as f:
    tags_set = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet101(pretrained=False)  # use the same model structure
num_ftrs = model.fc.in_features
num_classes = len(tags_set)
model.fc = nn.Linear(num_ftrs, num_classes)  # replace the last FC layer

# Load the trained model. A CUDA model won't load on CPU without the `map_location` parameter.
if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model = model.to(device)
model.eval()  # set model to evaluation mode


print('Reading from: ' + image_dir)
# Get image files in directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.JPG')]

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load tag whitelist
with open(whitelist_file, 'r') as f:
    whitelist = set(line.strip() for line in f)
    print(whitelist)

# Iterate over every image file
for image_file in image_files:
    # Load the image
    image_path = os.path.join(image_dir, image_file)
    print('Processing: ', image_path)
    input_image = Image.open(image_path).convert("RGB")

    # Transform the image
    input_image = transform(input_image).unsqueeze(0)

    # Move input to device and perform inference
    with torch.no_grad():
        input_image = input_image.to(device)
        outputs = model(input_image)

        # Convert the logits to probabilities
        sigmoids = torch.nn.Sigmoid()
        probabilities = sigmoids(outputs) * 100

    # for class_name, percentage in zip(tags_set, probabilities[0]):
    #     print('{}: {:.2f}%'.format(class_name, percentage.item()))

    # Apply sigmoid function and threshold at 0.5
    threshold = 0.5
    sigmoid = torch.nn.Sigmoid()
    predictions = sigmoid(outputs)
    predicted_classes = (predictions > threshold).squeeze()

    # Map binary predictions to class names
    if os.path.exists(whitelist_file):
        predicted_tags = [class_name for class_name, predicted_class in zip(tags_set, predicted_classes) if predicted_class == 1 and class_name in whitelist]
        print(predicted_tags)
        common_tags = set(predicted_tags) & whitelist
        print(common_tags)
    else:
        predicted_tags = [class_name for class_name, predicted_class in zip(tags_set, predicted_classes) if predicted_class == 1]

    # Save predictions into a text file named image_name.jpg.txt
    image_name = os.path.splitext(image_file)[0]
    file_name = f'{image_name}.txt'
    file_path = os.path.join(image_dir, file_name)
    with open(file_path, 'w') as f_out:
        for tag in predicted_tags:
            f_out.write(f'{tag}\n')

