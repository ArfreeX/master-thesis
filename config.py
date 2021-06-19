dataset = {
    'dataset': 'CarParts',  # {CarParts, Fruit35, Fruit262}

    'prepare_data': False,
    'split_dataset': False,
    'split_percentage': 20,
    'plot_example': False,

    'img_size': (160, 160),
    'batch_size': 20,

    'prefetch': True
}

model = {
    'model': 'VGG19',
    'weights': 'imagenet',
    'unfreeze_size': 15,
    'optimizer': 'SGD',
    'base_learning_rate': 0.0000001,
    'epochs': 10,
    'train': True,
    'save_model': False,
    'save_model_path': 'dataset_split_model_unfreezeSize_lr_epochs',
    'load_model': False,
    'load_model_path': 'Fruit35_10_EfficientNetB7_300_1e-05_3'
}

general = {
    'plot_train_history': False,
    'plot_test_sample_eval': False,
    'experiment_mode': True,
    'save_model_info': True,
    'save_plot_eval': True
}

experiment_setup = {
    # 'datasets': ['Fruit35', 'CarParts', 'Fruit262'],
    'datasets': ['CarParts'],
    'models': ['VGG19'],
    'unfreeze_sizes': [1, 2, 3, 4, 5, 7, 9, 11, 14, 17, 20, 23],
    'epochs': [10],
    'batch_sizes': [20],
    'lrs': [0.000001, 0.0000001]
}
