# CMPE258-Project
Deep Learning SJSU Course

## Contributors:
* Wil Carrasco

## Project Tile: 
Deep Learning CNN dog/cat Classifier w/VGG16
 
## Description:
Using a Convolutional Neural Network w/ VGG16 to identify cat & dog images.

## Project Idea
Using multiple images of cats and dogs can we create a CNN to categorize images between the two types.

## Proposed Solution
Step 1: Data ingestion, data cleaning, and initial visualization (if necessary)

Step 2: Preprocessing as needed - as we decide on an appropriate approach, this step may or may not be required

Step 3: Split dataset into training and testing datasets (80% training, 20% testing)

Step 4: Train the CNN model using the 80% of the dataset allocated to training

Step 5: Evaluate the model against the test set and verify accuracy metrics

Step 6: Revisit steps as required to incorporate learned techniques to improve metrics

Step 7: Feed the model new and unseen data and verify correct classification


Data Set
------------
[dogs-vs-cats Data Set](https://www.kaggle.com/biaiscience/dogs-vs-cats)  
[Download Link](https://www.kaggle.com/biaiscience/dogs-vs-cats/download)
=======

## Large File Storage & Pre-Build Models
Downloading these large files requires that [LFS](https://git-lfs.github.com/) is installed on your system.

Included is 'data.zip' which contains all the datasets used in this project. In addition, two pre-build models
are also included that can be used load load up and run immediate testing/usage since training takes 12+ hrs:

* 'cats_vs_dog_CNN_simple_85.h5' -- simple model with a 85% validation accuracy rating

* 'cats_vs_dog_CNN_VGG_98.h5' -- complex model using VGG16 with a 98.5% accuracy rating

model can be loaded with the following usage: https://www.tensorflow.org/guide/keras/save_and_serialize

```python
from tensorflow import keras

model = keras.models.load_model('path/to/location')
```
