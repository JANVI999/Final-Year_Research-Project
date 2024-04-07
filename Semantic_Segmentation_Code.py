pip install tensorflow
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive
# Function to load and preprocess images and masks
def load_data(image_folder, mask_folder, img_size=(256, 256)):
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
mask_paths = [os.path.join(mask_folder, mask) for mask in os.listdir(mask_folder)]
X = []
Y = []
for img_path, mask_path in zip(image_paths, mask_paths):
# Load and preprocess images
img = load_img(img_path, target_size=img_size)
img = img_to_array(img) / 255.0 # Normalize to [0, 1]
X.append(img)
              # Load and preprocess masks
mask = load_img(mask_path, target_size=img_size, color_mode="grayscale")
mask = img_to_array(mask) / 255.0 # Normalize to [0, 1]
Y.append(mask)
X = np.array(X)
Y = np.array(Y)
# Convert masks to categorical format
Y_categorical = to_categorical(Y, num_classes=2)
return X, Y_categorical
# Load data
images_folder_path = '/content/drive/MyDrive/AlgaeDetectionProject/training_images_folder'
masks_folder_path = '/content/drive/MyDrive/AlgaeDetectionProject/training_masks_folder'
X, Y_categorical = load_data(images_folder_path, masks_folder_path)
# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y_categorical, test_size=0.2, random_state=42)

def unet_model(input_size=(256, 256, 3), num_classes=2):
inputs = Input(input_size)
# Encoder
conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
# Bottom layer
conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
# Decoder
up1 = UpSampling2D(size=(2, 2))(conv4)
up1 = Conv2D(256, 2, activation='relu', padding='same')(up1)
merge1 = concatenate([conv3, up1], axis=-1)
conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge1)
conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
up2 = UpSampling2D(size=(2, 2))(conv5)
up2 = Conv2D(128, 2, activation='relu', padding='same')(up2)
merge2 = concatenate([conv2, up2], axis=-1)
conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge2)
conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
up3 = UpSampling2D(size=(2, 2))(conv6)
up3 = Conv2D(64, 2, activation='relu', padding='same')(up3)
merge3 = concatenate([conv1, up3], axis=-1)
conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge3)
conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
# Output layer
outputs = Conv2D(num_classes, 1, activation='softmax')(conv7)
model = Model(inputs=inputs, outputs=outputs)
return model
model = unet_model()
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=8, validation_data=(X_val, Y_val))
# Save the trained model
model.save('/content/drive/MyDrive/AlgaeDetectionProject/unet_model.h5')
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
# Function to load and preprocess test data
def load_test_data(image_folder, img_size=(256, 256)):
test_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
X_test = []
for img_path in test_paths:
# Load and preprocess test images
img = load_img(img_path, target_size=img_size)
img = img_to_array(img) / 255.0 # Normalize to [0, 1]
X_test.append(img)
X_test = np.array(X_test)
return X_test
# Load the trained model
model_path = '/content/drive/MyDrive/AlgaeDetectionProject/unet_model.h5'
model = load_model(model_path)
# Load test data
test_images_folder_path = '/content/drive/MyDrive/AlgaeDetectionProject/testing_images_folder'
X_test = load_test_data(test_images_folder_path)
# Make predictions
predictions = model.predict(X_test)
import cv2
from google.colab.patches import cv2_imshow
# Assuming binary classification, convert predictions to binary values (0 or 1)
binary_predictions = np.argmax(predictions, axis=-1)
# Visualize or save the predicted masks (you can adapt this based on your needs)
for i, prediction in enumerate(binary_predictions):
# Assuming you want to save the predicted masks
save_path = f'/content/drive/MyDrive/AlgaeDetectionProject/predicted_masks/{i}_mask.png'
mask_image = np.squeeze(prediction * 255).astype(np.uint8)
cv2.imwrite(save_path, mask_image)
# Alternatively, you can visualize the predicted masks using matplotlib or any other library
import matplotlib.pyplot as plt
# Assuming you want to visualize the first few predictions
for i in range(min(5, len(binary_predictions))):
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(X_test[i])
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(np.squeeze(binary_predictions[i]), cmap='gray')
plt.title('Predicted Mask')
plt.show()
              
