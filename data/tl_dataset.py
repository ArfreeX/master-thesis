import matplotlib.pyplot as plt
import os
import random
import shutil
import tensorflow as tf

from abc import ABC, abstractmethod

from tensorflow.keras.preprocessing import image_dataset_from_directory


class TLDataset(ABC):

    def __init__(self):
        super().__init__()

        self._train_dataset = None
        self._validation_dataset = None
        self._test_dataset = None
        self._categories = []

        self._dataset_dir = None
        self._train_dir = None
        self._validation_dir = None
        self._test_dir = None

    @abstractmethod
    def prepare_dataset(self):
        pass

    def load_data(self, img_size, batch_size):
        if not os.path.isdir(self._train_dir) and os.path.isdir(self._validation_dir) \
                and os.path.isdir(self._test_dir):
            raise Exception("Missing one of the required subdirectories: train, validation, test")

        for _, dirs, _ in os.walk(self._train_dir):
            for subdir in dirs:
                self._categories.append(subdir)

        self._train_dataset = image_dataset_from_directory(
            directory=self._train_dir,
            labels="inferred",
            label_mode="int",
            class_names=self._categories,
            image_size=img_size,
            batch_size=batch_size
        )

        self._validation_dataset = image_dataset_from_directory(
            directory=self._validation_dir,
            labels="inferred",
            label_mode="int",
            class_names=self._categories,
            image_size=img_size,
            batch_size=batch_size
        )

        self._test_dataset = image_dataset_from_directory(
            directory=self._test_dir,
            labels="inferred",
            label_mode="int",
            class_names=self._categories,
            image_size=img_size,
            batch_size=batch_size
        )

    def split_dataset(self, split_percentage):
        shutil.copytree(os.path.join(self._dataset_dir, 'pure'), self._train_dir)

        os.makedirs(self._validation_dir)
        os.makedirs(self._test_dir)

        for subdir in [f.path for f in os.scandir(self._train_dir) if f.is_dir()]:
            t_src = os.path.join(self._train_dir, os.path.basename(subdir))
            v_dest = os.path.join(self._validation_dir, os.path.basename(subdir))
            t_dest = os.path.join(self._test_dir, os.path.basename(subdir))

            if not os.path.exists(v_dest):
                os.makedirs(v_dest)
            if not os.path.exists(t_dest):
                os.makedirs(t_dest)

            class_size = os.listdir(subdir)
            v_t_size = len(class_size) * split_percentage // 100
            for i in range(v_t_size):
                file = random.choice(os.listdir(subdir))
                file_path = os.path.join(t_src, file)
                shutil.move(file_path, v_dest)

                file = random.choice(os.listdir(subdir))
                file_path = os.path.join(t_src, file)
                shutil.move(file_path, t_dest)

    def prefetch(self):
        autotune = tf.data.AUTOTUNE

        self._train_dataset = self._train_dataset.prefetch(buffer_size=autotune)
        self._validation_dataset = self._validation_dataset.prefetch(buffer_size=autotune)
        self._test_dataset = self._test_dataset.prefetch(buffer_size=autotune)

    def plot_sample(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self._train_dataset.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self._train_dataset.class_names[labels[i]])
                plt.axis("off")
        plt.show()

    def get_train(self):
        return self._train_dataset

    def get_validation(self):
        return self._validation_dataset

    def get_test(self):
        return self._test_dataset

    def get_categories(self):
        return self._categories
