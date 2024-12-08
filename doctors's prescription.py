# model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# Paths to image folder and labels file
image_folder = r"D:\python\Internship\project\Medicine name dataset\Training\training_words"
label_file = r"D:\python\Internship\project\Medicine name dataset\Training\training_labels.csv"

# Load and preprocess the data
def load_and_prepare_data(image_folder, label_file):
    labels_df = pd.read_csv(label_file)
    images = []
    labels = []

    for index, row in labels_df.iterrows():
        image_path = f"{image_folder}/{row['IMAGE']}"
        image = load_img(image_path, color_mode='grayscale', target_size=(64, 64, 1))
        image = img_to_array(image)
        images.append(image)
        labels.append(row['MEDICINE_NAME'])

    images = np.array(images, dtype="float") / 255.0
    labels = np.array(labels)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels, num_classes=len(label_encoder.classes_))

    return images, labels_categorical, label_encoder
    
# Load and prepare data
images, labels_categorical, label_encoder = load_and_prepare_data(image_folder, label_file)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential([
    Input(shape=(64, 64, 1)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the model
model.save('prescription_classification_model.keras')

# Save the LabelEncoder
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)