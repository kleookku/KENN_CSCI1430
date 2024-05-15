import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import AveragePooling2D, Conv2D, DepthwiseConv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD

hyperparams = {
    'learning_rate': 0.005,
    'epochs': 50,
    'dropout_rate': 0.3,
    'optimizer': 'sgd'
}

def build_discriminator(input_shape=(128, 128, 3), dropout_rate=0.3):
    model = Sequential([
        DepthwiseConv2D(kernel_size=3, padding='same', input_shape=input_shape, activation=swish),
        Conv2D(32, kernel_size=1),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),

        Conv2D(64, kernel_size=3, dilation_rate=2, padding='same', activation=swish),
        BatchNormalization(),
        AveragePooling2D(pool_size=2),

        Conv2D(128, kernel_size=3, padding='same', activation=swish),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),

        DepthwiseConv2D(kernel_size=3, padding='same', activation=swish),
        Conv2D(256, kernel_size=1),
        BatchNormalization(),
        AveragePooling2D(pool_size=2),

        Conv2D(512, kernel_size=3, padding='same', activation=swish),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),

        Flatten(),
        Dense(800, activation=swish),
        LeakyReLU(alpha=0.2),
        Dropout(dropout_rate),
        Dense(2, activation='softmax')
    ])
    return model

def train_discriminator():
    model = build_discriminator()
    
    optimizer = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_cb = ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_accuracy', verbose=1)
    early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy')
    reduce_lr_cb = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3, min_lr=0.0001, verbose=1)

    history = model.fit(
        train_dataset,
        epochs=hyperparams['epochs'],
        validation_data=val_dataset,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        batch_size=64
    )
    
    model.evaluate(test_dataset)
    
    return history

def visualization(model):
    images = []
    labels = []
    classifications = ["real", "fake"]
    for im in test_dataset:
        images.append(im[0])
        labels.append(im[1])
        break

    images = np.array(images)
    labels = np.array(labels)
    predictions = model.predict(im)
    plt.figure(figsize=(15, 10))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i])
        plt.title(f"True: {classifications[np.argmax(labels[i])]}, Predicted: {classifications[np.argmax(predictions[i])]}")
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    history = train_discriminator()

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

    visualization(model)
