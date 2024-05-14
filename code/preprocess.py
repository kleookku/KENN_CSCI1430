import tensorflow as tf

img_size = 128
b_size = 8

# data augmentation
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.1
)

# testing without data augmentation
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# training dataset
train_dataset = img_gen.flow_from_directory(
    '../data/rvf10k/train',
    target_size=(img_size, img_size),
    color_mode='rgb',
    class_mode='binary',
    batch_size=b_size,
    subset="training",
)

# validation dataset
val_dataset = img_gen.flow_from_directory(
    '../data/rvf10k/valid',
    target_size=(img_size, img_size),
    color_mode='rgb',
    class_mode='binary',
    batch_size=b_size,
    subset="validation",
)

# set up testing dataset
test_dataset = test_gen.flow_from_directory(
    '../data/rvf10k/valid',
    target_size=(img_size, img_size),
    color_mode='rgb',
    class_mode='binary',
    batch_size=b_size
)
