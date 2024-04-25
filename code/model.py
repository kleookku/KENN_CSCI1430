import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential

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

    return model

# Build the discriminator
discriminator = build_discriminator()

# Compile the model
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
discriminator.summary()
