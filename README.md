# Multitag-trainer
*Image classification made easy!*

## What is it? 
Multitag-trainer is a collection of Python scripts and notebooks created to make custom photo tagging models. The repo's key feature is the ability to assign multiple tags to a photo. 

## How does it work?
Many image classifiers require the images to be sorted into folders (e.g. "cats" and "dogs"). Multitag-trainer gets rid of that requirement and instead uses a system where images are given a text file containing the tags inside the images. 
This flexibility allows you to train simple monotag classifiers or complex feature detectors at will. 

Ready to give it a try? Then keep reading!

# Installation

## Environment setup
The recommended way to use Multitag-trainer is with [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and Python 3.11. 
Once installed, create your environment and give it a name. For example, 
```
conda create -n multitag-trainer python=3.11
```
Conda will then proceed to install the base packages. Use the following command to activate your environment:
```
conda activate multitag-trainer
```
and simply use `conda deactivate` once you're done. You know you've done it right if you see a `(base)` on the left of your username on Mac and Linux. Windows users might need to use the dedicated conda shell.

### Packages
Once conda is installed and set up, let's install the few packages we'll need. 
```
pip install scikit-learn python-dotenv
```

### Installing PyTorch
PyTorch is the machine learning library we'll use to train and run our model.  It runs on Windows, Mac (Intel and Apple Sillicon) and Linux.

To install PyTorch, go to [their website](https://pytorch.org/get-started/locally/).
If you have an Nvidia GPU on Windows or Linux, select the CUDA Compute Platform (check your CUDA version!) and follow the instructions. 
If you have a Mac, or a PC without an Nvidia GPU, select CPU / Default. 

> **IMPORTANT NOTE**: if you have installed PyTorch before via `pip install torch torchvision torchaudio`, then you may be running the CPU version, even if you have a CUDA GPU. 
> Run `pip uninstall torch, torchvision torchaudio` and `pip cache purge` before reinstalling with the command from PyTorch's website.

The scripts will tell you if you have CUDA enabled or not. If you want to check before running the scripts, open a python shell (just type `python` in your terminal) and use the following command: 
```python
import torch
torch.cuda.is_available()
```
If it returns 'True' then you're good to go. If 'False' and you have a recent Nvidia GPU, then check the note above. All other platforms will return "False", that's normal.

### Downloading the repo
You're almost there. All that's left is downloading and setting up the repo. Either download the code as zip or use
```
git clone https://github.com/xTellarin/multitag-trainer.git
```
in the folder of your choice.


# Setup
## Multitag-trainer
Alright, now it's time to prepare your training run. Start by telling Multitag-trainer where to find your files and customizing your model.

First, create a `.env` file to store your variables (note the dot before 'env'). Paste the following inside and make adjust to your liking:
```python
TRAINING=PATH/TO/TRAINING/FOLDER
INFERENCE_DIR=PATH/TO/INFERENCE/FOLDER
MODEL=MODEL/PATH/AND/NAME
TAGS=PATH/NAME/TO/PICKLE/FILE
```
Let's go through the file:
- **Training**: this is where your training files (images and tags) are. It can be an absolute or relative path. For example, if `training` is in the same location as my python files (the root directory), then `TRAINING=training/` 
  More info below.
- **Inference**: this is where the images that you want to tag are. Note that the model will only go through the images directly in the folder. If you have another folder within the inference folder, Multitag-trainer will ignore it.
- **Model**: this is where you want to save your model. You can create a `models/`  folder too! Remember to pass the path **and name** of the model you want to save. E.g.: `models/classifier.pth`.
- **Tags**: same as Model, but for the pickle (.pkl) file that will store all the tags the model learned. E.g.: `models/classifier_tags.pkl`


You can also create a whitelist.txt (optional, recommended if you train on a lot of tags). If you put a list of tags in there (one tag per line, no commas  `,`), the inference scripts will only tag your images with the tags you allowed in the whitelist. Example of a whitelist.txt:
```
tag1
tag2
tag3
...
```

## Training material
Now let's review the training material, or what your model will learn to recognize. 
As explained in the intro, the structure of the training folder is dead simple: put all your images (.jpg or .png) and an equal number of text files containing the tags. For example, if you have an `image1.jpg`, you need an `image1.jpg.txt` with tags following the same structure as the whitelist above.
Note that the `.txt` is **appended** to your full image filename, extension included. 
```
image1.jpg    image1.jpg.txt
image2.png    image2.png.txt
image3.jpg    image3.jpg.txt
```

If you have downloaded an existing dataset, or prepared your own, in a folder-based format, you can use the `prepare.py` or `prepare.ipynb` to add the same tag(s) to all the files in these folder, and then move both images and tags to your training folder.

If you don't have a set of images to train on yet, you can use the excellent [**gallery-dl**](https://github.com/mikf/gallery-dl) to download images and tags (on supported websites). The image / tags structure will be the same, you will only need to move the files to your training directory.

# Running
Now that your configuration and training files are ready, it's time to train your model.
You can use the `train.py` or `train.ipynb` scripts to do so. If you're unfamiliar with `ipynb` files, they're python notebook (Jupyter, Google Colab, even VS Code) extensions, and arguably the best way to do your test runs. 
Whichever you choose to use, locate the hyperparameters code block near the top of your file:
```python
# Hyperparameters  
num_epochs = 10  
batch_size = 32
```
You can use the default values to start, but you'll probably want to tweak at least the number of epochs once you get a baseline. Here are quick-and-dirty definitions to help you:
- **Epoch:** An epoch corresponds to one cycle through the full training dataset. In other words, it's a single pass through all of your training examples. For instance, if you have a training dataset of 10,000 images, after your model has seen all 10,000 images once, that's one epoch. The more epoch you train on, the more your model learns the features of your dataset. There are diminishing returns, and normally training too much can lead to overfitting, but there are guardrails in place to prevent that.
- **Batch size:** The batch size is the number of training samples processed before the model is updated. If you have a batch size of 100, it means the model updates its weights after it has seen 100 images. Essentially, batch size determines the number of samples to "show" to the model before updating it. Increasing batch size could lead to faster training, but there's a trade-off as it also requires more memory (GPU VRAM + RAM if CUDA, otherwise just RAM).

## Tips and FAQ
- **How many images should I use for training?**
	Preferably, as many as possible. That might be difficult in some cases, so make sure to have proper tagging and "quality" images. That doesn't always mean high resolution, it's more about 'is the subject clearly visible?'. Make sure to have more difficult images as well, or the model will have trouble identifying your subject(s) in complex images later. 
	You can make use of some tools to artificially fatten your training folder, such as flipping images along the vertical axis, or recoloring, cropping them, etc. But new images are always preferred.

	The answer also depends on what you're trying to identify. If it's cats vs dogs, you might see that 100 images or each or even less is enough. The more complex you want your model to be, the more training material you need. For a booru classifier, 20,000 images is a good value.
	
	If you're savvy enough, you can also change the base ResNet model used in the script for a bigger one, or a different CNN architecture if you'd like.
	
- **How do I know which epoch value or batch size to use?**
  There's no definitive answer here, you need to experiment. You can use the base values to get you started. Look at the validation loss, and when the script stops saving the model. The more material you have, generally the higher the number of epochs, but then also the longer your training takes. You can try to bump up the batch size, but mind your memory usage!
  For each epoch, the script will give you the number of true positives and negatives and false positives and negatives. You can calculate your F1 score and more advanced measures like PR or ROC curves. See [this article](https://towardsdatascience.com/various-ways-to-evaluate-a-machine-learning-models-performance-230449055f15) for more details.

# Using
Using the model, called inference (inferring?) is done through the two inference files. Note that the training notebook also has an inference part, but you need to have the model loaded in memory (right after training). 
Both files read from the model, tags and inference folder variables in your `.env` configuration file. In your inference folder, just place the images you want to tag and the model will try to assign tags. That's also how you can evaluate its performance (and missing tags).
The output will be the same as in the training folder, with a text file containing the tags and named after the image the model ran on.

In the notebook, locate the line `#print(probabilities)` and uncomment it to see the probabilities given to each tag. Be careful of using this if you have a lot of tags!
You can also use the last cell to get a list of every tag your model has learned.


Thanks for reading this far down and using Multitag-trainer! Feel free to contribute to the docs or code by submitting a PR request or creating a bug Issue or Discussion.

\- Tellarin
