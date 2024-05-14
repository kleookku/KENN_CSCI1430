# KENN_CSCI1430

## Overview
### Preprocessing
- Image Resizing
- Normalization
- Data augmentation
### Model construction
- Implemented convolutional neural network
### Visualization
- Displayed random examples of real versus fake images of faces
- Generated graphs to show the loss/accuracy of the model
### Prediction
- Loaded trained model from specific checkpoint
- Processed an image from given path to match input requirements Used the model to predict whether the image is 'Real' or 'Fake’ Displayed the image along with its predicted label.

## How to use
### Requirements
- Python 3.x
- Required libraries: tensorflow, matplotlib, numpy

### Training the Model
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/model-training.git
   cd model-training
2. Run the training script:
    ```sh
    python train.py