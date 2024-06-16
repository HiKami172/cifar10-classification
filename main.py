import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Define directories
DATA_DIR = "data/processed/"
MODEL_DIR = "models/"
REPORT_DIR = "reports/figures/"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


class CIFAR10Classifier:
    def __init__(self):
        self.model = None

    def load_and_preprocess_data(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        np.save(os.path.join(DATA_DIR, 'x_train.npy'), x_train)
        np.save(os.path.join(DATA_DIR, 'y_train.npy'), y_train)
        np.save(os.path.join(DATA_DIR, 'x_test.npy'), x_test)
        np.save(os.path.join(DATA_DIR, 'y_test.npy'), y_test)

        return (x_train, y_train), (x_test, y_test)

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model
        return model

    def train_model(self, x_train, y_train, x_test, y_test, batch_size=64, epochs=50):
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                 validation_data=(x_test, y_test), verbose=1)
        self.model.save(os.path.join(MODEL_DIR, 'cifar10_cnn_model.keras'))
        self.plot_training_history(history)

    def evaluate_model(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=1)
        print(f'Test Accuracy: {accuracy:.4f}')

        y_pred = np.argmax(self.model.predict(x_test), axis=1)
        y_true = np.argmax(y_test, axis=1)

        self.plot_confusion_matrix(y_true, y_pred)
        return accuracy

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.title('Model Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.title('Model Loss')

        plt.savefig(os.path.join(REPORT_DIR, 'training_history.png'))
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 10))
        cm_display = ConfusionMatrixDisplay(cm, display_labels=[
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'])
        cm_display.plot(cmap='Blues', ax=plt.gca())
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(REPORT_DIR, 'confusion_matrix.png'))
        plt.show()


if __name__ == "__main__":
    classifier = CIFAR10Classifier()
    (x_train, y_train), (x_test, y_test) = classifier.load_and_preprocess_data()
    model = classifier.create_model()
    classifier.train_model(x_train, y_train, x_test, y_test)
    classifier.evaluate_model(x_test, y_test)
