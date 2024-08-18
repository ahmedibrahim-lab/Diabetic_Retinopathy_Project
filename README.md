# APTOS 2019 Blindness Detection with Attention UNet

This repository contains the implementation of an Attention UNet model for detecting diabetic retinopathy from retinal images. The model is trained and validated on the APTOS 2019 dataset, which consists of labeled retinal images categorized into five classes.

## Dataset

The dataset used in this project is from the [APTOS 2019 Blindness Detection competition](https://www.kaggle.com/c/aptos2019-blindness-detection/data) on Kaggle. The dataset includes images of retina taken using fundus photography. The images are labeled based on the severity of diabetic retinopathy:

- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

### Dataset Structure

- `train.csv`: Contains the training data with columns `id_code` (image filename) and `diagnosis` (target label).
- `test.csv`: Contains the test data with only the `id_code` column.
- `train_images/`: Folder containing training images in PNG format.

## Model Architecture

The model is an Attention UNet, a modification of the standard UNet architecture, enhanced with attention gates that focus on relevant features in the input images. Key components of the model include:

- **DoubleConv:** A block consisting of two convolutional layers, each followed by batch normalization and ReLU activation.
- **AttentionGate:** Used to focus on the important parts of the input data by suppressing irrelevant regions.
- **UpConv:** Upsampling followed by a convolution layer.
- **Global Average Pooling and Classifier:** For final classification based on the features extracted by the UNet.

## Preprocessing

Several preprocessing steps were applied to the images:

1. **Background Removal:** Convert images to grayscale and remove background using a threshold to create a binary mask.
2. **Resizing:** Images are resized to 256x256 pixels.
3. **Normalization:** Images are normalized using ImageNet mean and standard deviation values.

## Training

### Data Augmentation

Data augmentation is applied to the training images to improve the model's generalization capability. The transformations include resizing, background removal, normalization, and more.

### Class Weights

Due to the class imbalance in the dataset, class weights are computed and used during training to penalize the model more for misclassifying underrepresented classes.

### Early Stopping

The model training process includes early stopping based on validation loss to prevent overfitting.

### Saliency Maps

Saliency maps are generated during training to visualize which parts of the images contribute most to the model's predictions.

## Evaluation Metrics

The following metrics are used to evaluate the model's performance:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

## Training Logs

Training is carried out for a maximum of 45 epochs with checkpoints saved after each epoch. The model performance is logged, including:

- Loss
- Accuracy
- Precision
- Recall
- F1 Score

## Results

After training, the model's performance is evaluated on the validation set using the metrics mentioned above. The training and validation metrics are plotted for analysis.

## Usage

- You can simply run the entire page in Google Colab.

### Acknowledgments

- The dataset was provided by the APTOS 2019 Blindness Detection competition on Kaggle.
- The architecture is inspired by the UNet model with attention mechanisms.