# Dog Breed Classifier using MobileNet from TensorFlow Hub

This is a Dog Breed Classifier that uses the MobileNet model from TensorFlow Hub to identify the breed of a dog from a given image. The model has been trained on a dataset with 120 different dog breeds.

## Dataset

The dataset used for training this classifier consists of labeled images of dogs belonging to different breeds. The dataset is split into training, validation, and test sets.

## Preprocessing

The images are preprocessed before being fed into the model. The preprocessing steps include:

- Reading the image file
- Converting the JPEG image to a tensor with three color channels
- Scaling the pixel values from the range [0-255] to [0-1]
- Resizing the image to a specified size (224x224 pixels in this case)

## Model Architecture

The classifier uses the MobileNet model from TensorFlow Hub as the base model. MobileNet is a lightweight convolutional neural network architecture designed for efficient inference on mobile and embedded devices. The MobileNet model is pretrained on the ImageNet dataset.

The classifier adds a dense layer with softmax activation as the output layer to predict the probabilities for each of the 120 dog breeds.

## Training

The model is trained using the training data with the following configuration:

- Loss function: Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy

The training is performed for a specified number of epochs, and early stopping is used to prevent overfitting.

## Evaluation

The model is evaluated on the validation set after each epoch to monitor its performance. The accuracy metric is used to measure the model's performance.

## Prediction

After training, the model is used to make predictions on the test set. The test images are preprocessed and fed into the model to obtain the predicted probabilities for each dog breed. The predictions are saved in a CSV file for submission.

## Usage

To use this Dog Breed Classifier, follow these steps:

1. Prepare the Dataset:

   - Split the dataset into training, validation, and test sets.
   - Ensure that the images are labeled with the corresponding dog breed.

2. Preprocess the Images:

   - Use the `preprocess_image` function to preprocess the images in the dataset.

3. Create Data Batches:

   - Use the `create_data_batches` function to create data batches from the preprocessed images and labels.
   - Specify the batch size and whether it's training, validation, or test data.

4. Build the Model:

   - Use the `create_model` function to build the Dog Breed Classifier model.
   - Specify the input shape, output shape (number of dog breeds), and the MobileNet model URL from TensorFlow Hub.

5. Train the Model:

   - Use the `train_model` function to train the model on the training data.
   - Specify the number of epochs for training.

6. Evaluate the Model:

   - View the model's performance on the validation set during training.
   - Monitor the accuracy metric to assess the model's performance.

7. Make Predictions:

   - After training, the model can be used to make predictions on new images.
   - Preprocess the new images using the `preprocess_image` function.
   - Use the trained model to predict the dog breed for each image.

8. Save and Load the Model:

   - The trained model can be saved and loaded for future use.
   - Use the `save_model` and `load_model` functions to save and load the model weights.

9. Submit Predictions:
   - Save the predictions on the test set in a CSV file for submission.

## Dependencies

The following dependencies are required to run the Dog Breed Classifier:

- TensorFlow
- TensorFlow Hub
- Pandas
- Matplotlib
