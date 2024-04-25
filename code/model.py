import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential

from preprocess import train_dataset, val_dataset, test_dataset

def build_discriminator(input_shape=(128, 128, 3)):
    model = Sequential([
        # Layer 1: Convolutional Layer
        Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),

        # Layer 2: Convolutional Layer
        Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),

        # Layer 3: Convolutional Layer
        Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),

        # Layer 4: Convolutional Layer
        Conv2D(512, (5, 5), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),

        # Flatten and Output Layer
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_discriminator():
    model = build_discriminator()
    model.summary()

    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset
    )

    print("Evaluating the model on test data:")
    scores = model.evaluate(test_dataset)
    print(f"Test Loss: {scores[0]}, Test Accuracy: {scores[1]}")

if __name__ == "__main__":
    train_discriminator()
