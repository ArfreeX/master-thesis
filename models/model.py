import os
import tensorflow as tf
import time

from pathlib import Path

from models.constants import CAR_PARTS_MODELS_DIR
from models.constants import FRUIT_262_MODELS_DIR
from models.constants import FRUIT_35_MODELS_DIR


def prepare_model(model_cfg, datasets):
    if model_cfg['model'] == 'EfficientNetB6':
        model_cfg['preprocess_input'] = tf.keras.applications.efficientnet.preprocess_input
        base_model = tf.keras.applications.EfficientNetB6(input_shape=model_cfg['img_shape'],
                                                          include_top=False,
                                                          weights=model_cfg['weights'])
    elif model_cfg['model'] == 'EfficientNetB7':
        model_cfg['preprocess_input'] = tf.keras.applications.efficientnet.preprocess_input
        base_model = tf.keras.applications.EfficientNetB7(input_shape=model_cfg['img_shape'],
                                                          include_top=False,
                                                          weights=model_cfg['weights'])
    else:
        raise Exception("Unknown model selected")

    model = Model(base_model, model_cfg, datasets)
    model.switch_trainable(model_cfg['unfreeze_size'])
    model.compile_model()

    training_time = 0
    if model_cfg['load_model']:
        load_path = model_cfg['load_model_path']
        if 'CarParts' in load_path:
            load_path = os.path.join(CAR_PARTS_MODELS_DIR, load_path)
        elif 'Fruit35' in load_path:
            load_path = os.path.join(FRUIT_35_MODELS_DIR, load_path)
        elif 'Fruit262' in load_path:
            load_path = os.path.join(FRUIT_262_MODELS_DIR, load_path)
        else:
            raise Exception("Invalid save model path provided")
        model.load_model(load_path)
        training_history = None
    elif model_cfg['train']:
        t = time.process_time()
        training_history = model.train_model()
        training_time = time.process_time() - t
    else:
        training_history = None

    if model_cfg['save_model']:
        Path(CAR_PARTS_MODELS_DIR).mkdir(parents=True, exist_ok=True)
        Path(FRUIT_35_MODELS_DIR).mkdir(parents=True, exist_ok=True)
        Path(FRUIT_262_MODELS_DIR).mkdir(parents=True, exist_ok=True)

        save_path = model_cfg['save_model_path']
        if 'CarParts' in save_path:
            save_path = os.path.join(CAR_PARTS_MODELS_DIR, save_path)
        elif 'Fruit35' in save_path:
            save_path = os.path.join(FRUIT_35_MODELS_DIR, save_path)
        elif 'Fruit262' in save_path:
            save_path = os.path.join(FRUIT_262_MODELS_DIR, save_path)
        else:
            raise Exception("Invalid save model path provided")
        model.save_model(save_path)

    return model.get_model(), training_history, training_time


class Model:
    def __init__(self, base_model, cfg, datasets):
        self.__base_model = base_model
        self.__cfg = cfg
        self.__datasets = datasets
        self.__model = None

    def compile_model(self):
        preprocess_input = self.__cfg['preprocess_input']

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(len(self.__datasets['categories']), activation='softmax')
        inputs = tf.keras.Input(shape=(160, 160, 3))
        x = preprocess_input(inputs)
        x = self.__base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)

        if self.__cfg['optimizer'] == 'SDG':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.__cfg['base_learning_rate'], momentum=0.8)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.__cfg['base_learning_rate'])
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        self.__model = model

    def switch_trainable(self, unfreeze_size):
        self.__base_model.trainable = True

        train_at = len(self.__base_model.layers) - unfreeze_size
        if train_at < 0:
            train_at = 0

        for layer in self.__base_model.layers[:train_at]:
            layer.trainable = False
        # self.__base_model.summary()

    def train_model(self):
        return self.__model.fit(self.__datasets['train'],
                                epochs=self.__cfg['epochs'],
                                validation_data=self.__datasets['validation'])

    def load_model(self, path):
        self.__model = tf.keras.models.load_model(path)

    def get_model(self):
        return self.__model

    def save_model(self, path):
        self.__model.save(path)
