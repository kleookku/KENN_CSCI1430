import tensorflow as tf
import tensorflow as tf

img_size = 128
b_size = 64  

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale  = 1./255,
                            zoom_range = .1,
                            horizontal_flip=True,
                            brightness_range= (0.8,1.2),
                            validation_split = .1)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale  = 1./255)

train_dataset = img_gen.flow_from_directory(
    '../data/rvf10k/train',
    target_size=(img_size, img_size),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=b_size,
    subset="training",
)

val_dataset = img_gen.flow_from_directory(
    '../data/rvf10k/train',
    target_size=(img_size, img_size),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=b_size,
    subset="validation",
)

test_dataset = test_gen.flow_from_directory(
    '../data/rvf10k/valid',
    target_size=(img_size, img_size),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=b_size,
    #subset=None,
)