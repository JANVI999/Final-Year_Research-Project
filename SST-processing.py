import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Assuming you have a function for loading a single image
def load_single_image(image_path):
    # Use cv2.imread or any other method to load your image
    img = cv2.imread(image_path)
    # Resize the image to match the model's expected input shape (256, 256, 3)
    img = cv2.resize(img, (256, 256))
    # Convert BGR to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def input_processing(image_path):
    # Use cv2.imread or any other method to load your image
    img = cv2.imread(image_path)
    # Resize the image to match the model's expected input shape (256, 256, 3)
    img = cv2.resize(img, (256, 256))
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Assuming you have a model for predictions
# Replace this with your actual model and preprocessing steps
def predict_mask(model, image):
    # Add any necessary preprocessing steps here
    # For example, normalization, etc.
    # Ensure that the input shape matches your model's expectations

    # Assuming model.predict takes a batch of images, you can add a batch dimension
    image = np.expand_dims(image, axis=0)

    # Make predictions
    predictions = model.predict(image)

    # Assuming binary classification, convert predictions to binary values (0 or 1)
    binary_predictions = np.argmax(predictions, axis=-1)

    return np.squeeze(binary_predictions)

# Specify the paths
image_path = 'D:\ChlaAndSST\image.png'
ground_truth_path = 'D:\ChlaAndSST\mask.png'

# Load the single image
in_image = input_processing(image_path)
input_image = load_single_image(image_path)

# Load the ground truth mask
ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

# Load the trained model
model_path = 'D:\ChlaAndSST\SST_model.h5'
model = load_model(model_path)

# Get the predicted mask for the single image
predicted_mask = predict_mask(model, input_image)

# plt.subplot(1, 3, 3)
# plt.imshow(predicted_mask.astype('uint8'), cmap='gray')  # Ensure mask data type is uint8
# plt.title('Predicted Mask')

plt.show()
