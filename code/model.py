import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import numpy as np
from preprocess import train_dataset, val_dataset, test_dataset

# Hyperparameters
hyperparams = {
    'learning_rate': 0.005,
    'epochs': 10,
    'dropout_rate': 0.3,
    'optimizer': 'adam'
}

def build_discriminator(input_shape=(128, 128, 3), dropout_rate=0.3):
    model = Sequential([
        Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape),
        LeakyReLU(alpha=0.2),
        Dropout(dropout_rate),
        Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(dropout_rate),
        Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(dropout_rate),
        Conv2D(512, (5, 5), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(dropout_rate),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    if hyperparams['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=hyperparams['learning_rate'])
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=hyperparams['learning_rate'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model
def show_random_images(dataset, title, num_images=2):
    plt.figure(figsize=(10, 2))
    plt.suptitle(title)
    for i in range(num_images):
        batch = next(iter(dataset))
        image = batch[0][i]  # Image data
        label = batch[1][i]  # Corresponding label
        
        label = 'Real' if label > 0.5 else 'Fake'
        ax = plt.subplot(1, num_images, i + 1)
        ax.imshow(image)
        ax.set_title(label)
        ax.axis('off')
    plt.show()

def predict_and_visualize(model_path, dataset, num_samples=10):
    model = tf.keras.models.load_model(model_path)
    plt.figure(figsize=(15, 10))
    count = 0
    for images, labels in dataset:
        for i in range(len(images)):
            img = images[i]
            label = labels[i]
            img_array = np.expand_dims(img, axis=0)
            pred = model.predict(img_array)
            pred_label = 'Real' if pred > 0.5 else 'Fake'
            true_label = 'Real' if label > 0.5 else 'Fake'
            ax = plt.subplot(num_samples // 2, 4, count + 1)
            ax.imshow(img)
            ax.set_title(f"True Label: {true_label} | Predicted Label: {pred_label}")
            ax.axis("off")
            count += 1
            if count == num_samples:
                plt.tight_layout()
                plt.show()
                return  # Exit after processing the desired number of samples
            
def train_discriminator():
    model = build_discriminator()
    model.summary()

    # the call back ensures it will save the best model 
    checkpoint_cb = ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_accuracy', verbose=1)

    # the actual call to train
    history = model.fit(
        train_dataset,
        epochs=hyperparams['epochs'],
        validation_data=val_dataset,
        callbacks=[checkpoint_cb]
    )

    

if __name__ == "__main__":
    # here we are just doing some sanity check, can honestly just comment out
    show_random_images(train_dataset, "Random Training Images")
    show_random_images(val_dataset, "Random Validation Images")

    # main call
    train_discriminator()

    # test and other
    print("Evaluating the model on test data:")
    model = tf.keras.models.load_model("best_model.keras")
    scores = model.evaluate(test_dataset)
    print(f"Test Loss: {scores[0]}, Test Accuracy: {scores[1]}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.show()

    predict_and_visualize("best_model.keras", test_dataset, num_samples=10)
