import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_model(model_path):
    """Load a saved Keras model."""
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess an image."""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    return img_array

def predict_image(model, img_array):
    """Predict the label of a single image using a loaded model."""
    prediction = model.predict(img_array)
    return 'Real' if prediction > 0.5 else 'Fake'

def display_image(image_path, prediction):
    """Display the image and the prediction."""
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title(f"Predicted Label: {prediction}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    model_path = "best_model.keras"  # Path to the best model checkpoint
    image_path = "path_to_your_image.jpg"  # Path to the image for prediction

    model = load_model(model_path)
    img_array = preprocess_image(image_path)
    prediction = predict_image(model, img_array)
    display_image(image_path, prediction)
