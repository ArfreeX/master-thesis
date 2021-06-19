import matplotlib.pyplot as plt
import numpy as np

from tensorflow.python.keras.utils.layer_utils import count_params

import config as cfg

from data.data_prep import prepare_data
from helpers.result_gathering import gather_results
from models.model import prepare_model


class ExperimentControl:
    def __init__(self, config):
        self.__cfg = config
        self.__experiment_mode = config.general['experiment_mode']
        self.__datasets = config.experiment_setup['datasets']
        self.__models = config.experiment_setup['models']
        self.__unfreeze_sizes = config.experiment_setup['unfreeze_sizes']
        self.__epochs = config.experiment_setup['epochs']
        self.__batch_sizes = config.experiment_setup['batch_sizes']
        self.__lrs = config.experiment_setup['lrs']

    def run_experiment(self):
        if not self.__experiment_mode:
            ExperimentControl.run_session(self.__cfg, True)
        else:
            for data in self.__datasets:
                self.__cfg.dataset['dataset'] = data
                for model in self.__models:
                    self.__cfg.model['model'] = model
                    for unfreeze_size in self.__unfreeze_sizes:
                        self.__cfg.model['unfreeze_size'] = unfreeze_size
                        for epoch_val in self.__epochs:
                            if unfreeze_size != 0 and epoch_val != 10:
                                continue
                            self.__cfg.model['epochs'] = epoch_val
                            for batch_size in self.__batch_sizes:
                                self.__cfg.dataset['batch_size'] = batch_size
                                for lr in self.__lrs:
                                    self.__cfg.model['base_learning_rate'] = lr
                                    print(
                                        f"Dataset: {data}, model: {model}, unfreeze_size: {unfreeze_size}, epoch_val: "
                                        f"{epoch_val}, batch_size: {batch_size}, lr: {lr}")
                                    ExperimentControl.run_session(self.__cfg, False)

    @staticmethod
    def run_session(exp_cfg, short_result):
        exp_dataset = prepare_data(exp_cfg.dataset)
        exp_cfg.model['img_shape'] = exp_cfg.dataset['img_size'] + (3,)
        exp_cfg.model['save_model_path'] = f"{exp_cfg.dataset['dataset']}_{exp_cfg.dataset['split_percentage']}_" \
                                           f"{exp_cfg.model['model']}_{exp_cfg.model['unfreeze_size']}_" \
                                           f"{exp_cfg.dataset['batch_size']}_{exp_cfg.model['epochs']}_" \
                                           f"{format(exp_cfg.model['base_learning_rate'], '.12f')}"

        exp_model, history, training_time = prepare_model(exp_cfg.model, exp_dataset)

        trainable_params = count_params(exp_model.trainable_weights)
        test_loss, test_accuracy = exp_model.evaluate(exp_dataset['test'])

        exp_model_eval = {
            'loss': test_loss,
            'acc': test_accuracy,
            'history': history,
            'time': training_time,
            'params': trainable_params
        }
        gather_results(exp_cfg.dataset, exp_cfg.model, exp_cfg.general, exp_model_eval)

        if exp_cfg.general['plot_test_sample_eval']:
            ExperimentControl.plot_test_example(exp_model, exp_dataset)

        if short_result:
            print(f"Trained params: {trainable_params}")
            print(f"Training time: {training_time} seconds")
            print(f"Model accuracy: {test_accuracy}")
            print(f"Model loss: {test_loss}")
            print(f"Dataset config: {exp_cfg.dataset}")
            print(f"Model config: {exp_cfg.model}")

    @staticmethod
    def plot_test_example(model, dataset):
        plt.figure(figsize=(10, 10))
        for images, labels in dataset['test'].take(1):
            for i in range(9):
                prediction = model.predict(np.expand_dims(images[i], axis=0))
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(dataset['categories'][prediction[0].argmax(axis=-1)] + " " + str(max(prediction[0])))
                plt.axis("off")
        plt.show()


if __name__ == '__main__':
    experiment = ExperimentControl(cfg)
    experiment.run_experiment()
