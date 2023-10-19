# Pneumonia X-ray Image Classification

This repository contains a dataset of pneumonia X-ray images with four categories: Pneumonia Bacterial, Pneumonia Viral, Normal, and Covid-19. The dataset consists of 9208 images, which are split into a training set, a testing set, and a validation set. A convolutional neural network (CNN) model was developed to classify these X-ray images.

## Dataset

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/artyomkolas/3-kinds-of-pneumonia). It consists of X-ray images categorized into four classes:

1. Pneumonia Bacterial
2. Pneumonia Viral
3. Normal
4. Covid-19

## Dataset Split
The dataset was divided as follows:
- Training set: 80% of the total dataset
- Testing set: 10% of the total dataset
- Validation set: 10% of the total dataset

## Data Augmentation
To improve model generalization, data augmentation techniques were applied using the `ImageDataGenerator` from Keras. This technique artificially increases the diversity of the training dataset by applying random transformations to the images, such as rotation, zooming, and flipping.

## Model
Several pre-trained models were experimented with, including ResNet-50, EfficientNetB3, EfficientNetB4, and MobileNetV2. After experimenting, MobileNetV2 was chosen as it provided the best performance.

The MobileNetV2 model was modified by adding three additional fully connected (Dense) layers with 512, 512, and 128 neurons, respectively. The final output layer was configured for multi-class classification with four categories.

## Training
The model was trained using the following configurations:
- Batch size: 32
- Input image size: 224x224 pixels
- Number of epochs: 15
- Optimizer: Adam

After training, the model achieved the following accuracy results:
- Training accuracy: 86%
- Testing accuracy: 84%

## Model Saving
The trained model was saved using the `h5` file format.

## Model Evaluation
The model was evaluated on the validation set, and it achieved an accuracy of 80.66%.

## Modules Used
The following Python modules were used in this project:
- `os`: For file and directory operations
- `shutil`: For file manipulation
- `random`: For random data sampling
- `tensorflow`: For building and training the neural network
- `matplotlib`: For data visualization
- `numpy`: For numerical operations
- `PIL`: For image processing
- `ImageDataGenerator`: For data augmentation
- `MobileNetV2`: A pre-trained deep learning model
- `load_model`: For saving and loading the trained model
- `accuracy_score`: From scikit-learn to measure model accuracy

## Usage
You can use the provided code and model to classify X-ray images into the specified categories. Please refer to the Python script and Jupyter notebook provided in this repository for more details.

## License
This dataset is provided for educational and research purposes. Please check the original data source for any specific licensing or usage terms.

---

**Note**: Make sure to install the required Python libraries before running the code. You can use `pip` to install the necessary packages:

```bash
pip install tensorflow matplotlib numpy pillow scikit-learn
```

Remember to also download and place the dataset in the appropriate directory as specified in the code.
