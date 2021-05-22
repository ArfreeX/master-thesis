import matplotlib.pyplot as plt

import numpy as np
import os
import tensorflow as tf
import random
import shutil

from tensorflow.keras.preprocessing import image_dataset_from_directory


_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.getcwd(), 'datasets/veggies_and_fruits')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

PREP_VEGGIE = False

if PREP_VEGGIE:
    VAL_TEST_SIZE = 10

    for subdir in [f.path for f in os.scandir(train_dir) if f.is_dir()]:
        val_dest = os.path.join(validation_dir, os.path.basename(subdir))
        test_dest = os.path.join(test_dir, os.path.basename(subdir))
        train_src = os.path.join(train_dir, os.path.basename(subdir))
        print(val_dest)
        print(test_dest)
        if not os.path.exists(val_dest):
            os.makedirs(val_dest)
        if not os.path.exists(test_dest):
            os.makedirs(test_dest)

        for i in range(VAL_TEST_SIZE):
            file = random.choice(os.listdir(subdir))
            file_path = os.path.join(train_src, file)
            shutil.move(file_path, val_dest)

            file = random.choice(os.listdir(subdir))
            file_path = os.path.join(train_src, file)
            shutil.move(file_path, test_dest)

categories = []
for _, dirs, _ in os.walk(train_dir):
    for subdir in dirs:
        categories.append(subdir)

BATCH_SIZE = 20
IMG_SIZE = (160, 160)

train_dataset = image_dataset_from_directory(
    directory=train_dir,
    labels="inferred",
    label_mode="categorical",
    class_names=categories,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


categories = []
for _, dirs, _ in os.walk(validation_dir):
    for subdir in dirs:
        categories.append(subdir)

validation_dataset = image_dataset_from_directory(
    directory=validation_dir,
    label_mode="categorical",
    labels="inferred",
    class_names=categories,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


categories = []
for _, dirs, _ in os.walk(test_dir):
    for subdir in dirs:
        categories.append(subdir)

test_dataset = image_dataset_from_directory(
    directory=test_dir,
    labels="inferred",
    label_mode="categorical",
    class_names=categories,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


class_names = train_dataset.class_names
print(class_names)

PLOT_EXAMPLE = False
if PLOT_EXAMPLE:
    plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

print('Number of train batches: %d' % tf.data.experimental.cardinality(train_dataset))
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(len(categories), activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


inputs = tf.keras.Input(shape=(160, 160, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


initial_epochs = 10

loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))


history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()