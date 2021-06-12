import matplotlib.pyplot as plt
import numpy as np

from tensorflow.python.keras.utils.layer_utils import count_params

import config as cfg

from data.data_prep import prepare_data
from helpers.result_gathering import gather_results
from models.model import prepare_model


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


def run_experiment(exp_cfg, short_result):
    exp_dataset = prepare_data(exp_cfg.dataset)
    exp_cfg.model['img_shape'] = exp_cfg.dataset['img_size'] + (3,)
    exp_cfg.model['save_model_path'] = f"{exp_cfg.dataset['dataset']}_{exp_cfg.dataset['split_percentage']}_" \
                                       f"{exp_cfg.model['model']}_{exp_cfg.model['unfreeze_size']}_" \
                                       f"{exp_cfg.dataset['batch_size']}_{exp_cfg.model['epochs']}_" \
                                       f"{format(exp_cfg.model['base_learning_rate'], '.12f')}"

    model, history, training_time = prepare_model(exp_cfg.model, exp_dataset)

    trainable_params = count_params(model.trainable_weights)
    test_loss, test_accuracy = model.evaluate(exp_dataset['test'])

    model_eval = {
        'loss': test_loss,
        'acc': test_accuracy,
        'history': history,
        'time': training_time,
        'params': trainable_params
    }
    gather_results(exp_cfg.dataset, exp_cfg.model, exp_cfg.general, model_eval)

    if short_result:
        print(f"Trained params: {trainable_params}")
        print(f"Training time: {training_time} seconds")
        print(f"Model accuracy: {test_accuracy}")
        print(f"Model loss: {test_loss}")
        print(f"Dataset config: {exp_cfg.dataset}")
        print(f"Model config: {exp_cfg.model}")


if __name__ == '__main__':
    if cfg.general['experiment_mode']:
        datasets = cfg.experiment_setup['datasets']
        models = cfg.experiment_setup['models']
        unfreeze_sizes = cfg.experiment_setup['unfreeze_sizes']
        epochs = cfg.experiment_setup['epochs']
        batch_sizes = cfg.experiment_setup['batch_sizes']
        lrs = cfg.experiment_setup['lrs']

        for data in datasets:
            cfg.dataset['dataset'] = data
            for model in models:
                cfg.model['model'] = model
                for unfreeze_size in unfreeze_sizes:
                    cfg.model['unfreeze_size'] = unfreeze_size
                    for epoch_var in epochs:
                        if unfreeze_size != 0 and epoch_var != 10:  # Pre-overfitting solution lock
                            continue
                        cfg.model['epochs'] = epoch_var
                        for batch_size in batch_sizes:
                            cfg.dataset['batch_size'] = batch_size
                            for lr in lrs:
                                cfg.model['base_learning_rate'] = lr
                                print(f"Dataset: {data}, model: {model}, unfreeze_size: {unfreeze_size}, epoch_var: "
                                      f"{epoch_var}, batch_size: {batch_size}, lr: {lr}")
                                run_experiment(cfg, False)
    else:
        run_experiment(cfg, True)

