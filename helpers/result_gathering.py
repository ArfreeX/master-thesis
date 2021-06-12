import csv
import matplotlib.pyplot as plt
import os

from pathlib import Path

from helpers.constants import MAIN_CSV_PATH
from helpers.constants import PLOTS_DIR
from helpers.constants import RESULTS_DIR


def gather_results(dataset_cfg, model_cfg, general_cfg, model_eval):
    rg = ResultGatherer(dataset_cfg, model_cfg, model_eval)

    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    if general_cfg['save_model_info']:
        rg.save_results_to_csv(MAIN_CSV_PATH)

    if general_cfg['save_plot_eval']:
        rg.save_train_history_plot(os.path.join(PLOTS_DIR, model_cfg['save_model_path']))


class ResultGatherer:
    def __init__(self, data_cfg, model_cfg, model_eval):
        self.__dataset = data_cfg['dataset']
        self.__split = data_cfg['split_percentage']
        self.__batch_size = data_cfg['batch_size']

        self.__model = model_cfg['model']
        self.__unfreeze = model_cfg['unfreeze_size']
        self.__lr = model_cfg['base_learning_rate']
        self.__epochs = model_cfg['epochs']

        self.__test_loss = model_eval['loss']
        self.__test_acc = model_eval['acc']
        self.__history = model_eval['history']
        self.__training_time = model_eval['time']
        self.__params = model_eval['params']

    def save_results_to_csv(self, path):
        print(path)
        with open(path, 'a') as file:
            fieldnames = ['dataset', 'dataset_split', 'model', 'unfreezed_layers', 'epochs', 'batch_size', 'lr',
                          'test_loss', 'test_acc', 'training_time', 'trained_parameters']
            data_writer = csv.DictWriter(file, fieldnames=fieldnames)
            print(path)
            if file.tell() == 0:
                data_writer.writeheader()
            data_writer.writerow({'dataset': self.__dataset, 'dataset_split': self.__split,
                                  'model': self.__model, 'unfreezed_layers': self.__unfreeze, 'epochs': self.__epochs,
                                  'batch_size': self.__batch_size, 'lr': self.__lr, 'test_loss': self.__test_loss,
                                  'test_acc': self.__test_acc, 'training_time': self.__training_time,
                                  'trained_parameters': self.__params})

    def save_train_history_plot(self, path):
        history = self.__history
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
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 2.5])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.savefig(path + '.png')
        plt.close()
