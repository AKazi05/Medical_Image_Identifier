import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Define the paths
image_dir = '/Users/akibk/Downloads/Medical_Image_Identifier/images'
labels_path = '/Users/akibk/Downloads/Medical_Image_Identifier/Data_Entry_2017_v2020.csv'

# Check if paths are correct
if not os.path.isdir(image_dir):
    raise FileNotFoundError(f"Image directory not found: {image_dir}")
if not os.path.isfile(labels_path):
    raise FileNotFoundError(f"Labels file not found: {labels_path}")

# Load labels
labels_df = pd.read_csv(labels_path)

# Preprocess labels
labels_df['Finding Labels'] = labels_df['Finding Labels'].apply(lambda x: x.split('|'))
labels_df = labels_df.explode('Finding Labels')

# Filter the labels DataFrame to only include rows for which images exist
labels_df = labels_df[labels_df['Image Index'].isin(os.listdir(image_dir))]

# Encode labels
mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(labels_df.groupby('Image Index')['Finding Labels'].apply(list))

# Add encoded labels back to the DataFrame
labels_df = labels_df.drop_duplicates(subset=['Image Index'])
labels_df['Encoded Labels'] = list(encoded_labels)

# Sample a subset of images (500 images)
subset_df = labels_df.sample(n=500, random_state=42)

# Ensure the same number of images and labels
def load_images(img_dir, labels_df, img_size=(224, 224)):
    images = []
    for img_name in labels_df['Image Index']:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
    return np.array(images)

# Load the images that match the subset
subset_images = load_images(image_dir, subset_df)
subset_labels = np.array(list(subset_df['Encoded Labels']))

# Ensure consistency between images and labels
assert len(subset_images) == len(subset_labels), "Number of images and labels do not match."

# Split data
X_train, X_test, y_train, y_test = train_test_split(subset_images, subset_labels, test_size=0.2, random_state=42)

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add a channel dimension
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Convert grayscale to RGB by repeating the channels
X_train = np.repeat(X_train, 3, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)

# Load a pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(mlb.classes_), activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Train the model
batch_size = 32
epochs = 10

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test)
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Predict on test data
y_pred = model.predict(X_test)

# Calculate metrics
y_test_binary = (y_test > 0.5).astype(int)
y_pred_binary = (y_pred > 0.5).astype(int)

precision = precision_score(y_test_binary, y_pred_binary, average='weighted')
recall = recall_score(y_test_binary, y_pred_binary, average='weighted')
f1 = f1_score(y_test_binary, y_pred_binary, average='weighted')
roc_auc = roc_auc_score(y_test_binary, y_pred, average='weighted', multi_class='ovr')

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'ROC-AUC: {roc_auc:.2f}')

# Save the model
model.save('chest_xray_model2.keras')

# Function to preprocess new images
def preprocess_image(img_path, img_size=(224, 224)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img
