
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import itertools
import warnings
from scipy.io import loadmat
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin


warnings.filterwarnings('ignore')


# Loading Datasets
def load_data_and_labels(data, labels, n = 10000):
    data_df = pd.read_csv(data, header=None, nrows = n)
    X = data_df.values.reshape((-1,28,28,4)).clip(0,255).astype(np.uint8)
    labels_df = pd.read_csv(labels, header=None, nrows = n)
    Y = labels_df.values.getfield(dtype=np.int8)
    return X, Y

# Loading SAT 6 dataset
X_train, y_train = load_data_and_labels(data='/content/drive/My Drive/X_train_sat6.csv',
                                        labels='/content/drive/My Drive/y_train_sat6.csv')
X_test, y_test = load_data_and_labels(data='/content/drive/My Drive/X_test_sat6.csv',
                                      labels='/content/drive/My Drive/y_test_sat6.csv')

# Loading Sat 4 Dataset
# X_train, y_train = load_data_and_labels(data='/content/drive/My Drive/X_train_sat4.csv',
#                                         labels='/content/drive/My Drive/y_train_sat4.csv')
# X_test, y_test = load_data_and_labels(data='/content/drive/My Drive/X_test_sat4.csv',
#                                       labels='/content/drive/My Drive/y_test_sat4.csv')


# Implementing CNN 
cnn_model = keras.Sequential([

    # First Convolutional Block
    layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same', input_shape=(28,28,4)),
    layers.MaxPool2D(),
    layers.Dropout(0.3),

    # Second Convolutional Block
    layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same',strides=2, ),
    layers.Dropout(0.3),

    # Third Convolutional Block
    layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same', strides=2,),
    layers.Dropout(0.3),

    # Fourth Convolutional Block
    layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding='same', ),
    layers.MaxPool2D(),
    layers.Dropout(0.3),

    # Classifier Head
    layers.Flatten(),
    layers.Dense(units=32, activation="relu"),
    layers.Dense(units=6, activation='softmax'),
])

cnn_model.summary()

# Compile model
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
cnn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

cnn_model_history = cnn_model.fit(X_train, y_train, batch_size=200, epochs=20, verbose=1, validation_data=(X_test, y_test))

# Model  Results
history_frame = pd.DataFrame(cnn_model_history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot(title = 'CNN Model Loss Curves')
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot(title = 'CNN Model Accuracy Curves');

print(classification_report(test_Y, pred_Y))
print('Overall Accuracy: %2.2f%%' % (100*accuracy_score(test_Y, pred_Y)))

# Implementation of DBN

# Preprocess the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 28*28*4))
X_test_scaled = scaler.transform(X_test.reshape(-1, 28*28*4))

# Define the DBN model
dbn_model = Pipeline(steps=[
    ('rbm', BernoulliRBM(n_components=100, n_iter=20, learning_rate=0.01, verbose=1)),
    ('logistic', LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto'))
])

# Ensure the scaled data has the same number of rows as the original data
assert X_train.shape[0] == X_train_scaled.shape[0], "Mismatch in number of rows between X_train and X_train_scaled"
assert X_test.shape[0] == X_test_scaled.shape[0], "Mismatch in number of rows between X_test and X_test_scaled"

min_length = min(X_train_scaled.shape[0], y_train.shape[0])
X_train_scaled = X_train_scaled[:min_length]
y_train = y_train[:min_length]

dbn_model.fit(X_train_scaled, y_train)

min_length = min(X_test_scaled.shape[0], y_test.shape[0])
X_test_scaled = X_test_scaled[:min_length]
y_test = y_test[:min_length]

# Evaluate the DBN model
dbn_accuracy = dbn_model.score(X_test_scaled, y_test)
print("DBN Test Accuracy:", dbn_accuracy*100)


# Implementation of DenseNet121
class LoadAndPreprocessData(BaseEstimator, TransformerMixin):
    def __init__(self, batch_size=32, target_size=(224, 224)):
        self.batch_size = batch_size
        self.target_size = target_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data_df, labels_df = X
        data_generator = self._data_generator(data_df, labels_df)
        return data_generator

    def _data_generator(self, data_df, labels_df):
        num_samples = len(data_df)

        while True:
            shuffled_indices = np.random.permutation(num_samples)
            for batch_idx in range(0, num_samples, self.batch_size):
                batch_indices = shuffled_indices[batch_idx:batch_idx+self.batch_size]
                batch_data = data_df.iloc[batch_indices].values.reshape((-1, 28, 28, 4))
                batch_labels = labels_df.iloc[batch_indices].values
                batch_images = self._preprocess_batch(batch_data)
                yield batch_images, batch_labels

    def _preprocess_batch(self, batch_data):
        processed_images = []
        for img in batch_data:
            img = array_to_img(img)
            img = img.resize(self.target_size)
            img = img_to_array(img)
            processed_images.append(img)

        X_processed = np.stack(processed_images)
        X_processed = preprocess_input(X_processed)
        return X_processed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Example usage
X_train_path = '/content/drive/My Drive/X_train_sat6.csv'
y_train_path = '/content/drive/My Drive/y_train_sat6.csv'
X_test_path = '/content/drive/My Drive/X_test_sat6.csv'
y_test_path = '/content/drive/My Drive/y_test_sat6.csv'

# Define pipeline
pipeline = Pipeline([
    ('loader', LoadAndPreprocessData(batch_size=128)),
])

# Load and preprocess data using generator
train_data_generator = pipeline.transform((pd.read_csv(X_train_path, header=None), pd.read_csv(y_train_path, header=None)))
test_data_generator = pipeline.transform((pd.read_csv(X_test_path, header=None), pd.read_csv(y_test_path, header=None)))


def build_model(num_classes):
    base_model = DenseNet121(include_top=False, input_shape=(224, 224, 4), weights=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train the model
model = build_model(num_classes=6)
history = model.fit(train_data_generator, steps_per_epoch=100000//128, epochs=20, validation_data=test_data_generator, validation_steps=1000//128)

# Plot accuracy and loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate the model on test set
test_loss, test_accuracy = model.evaluate(test_data_generator, steps=1000//128, verbose=2)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

