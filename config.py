dataset = {
    'dataset': 'Fruit35',  # {CarParts, Fruit35, Fruit262}

    'prepare_data': False,
    'split_dataset': False,
    'split_percentage': 10,
    'plot_example': False,

    'img_size': (160, 160),
    'batch_size': 32,

    'prefetch': True
}

model = {
    'model': 'EfficientNetB7',
    'weights': 'imagenet',
    'unfreeze_size': 300,
    'optimizer': 'Adam',
    'base_learning_rate': 0.0000001,
    'epochs': 100,
    'train': True,
    'save_model': False,
    'save_model_path': 'dataset_split_model_unfreezeSize_lr_epochs',
    'load_model': False,
    'load_model_path': 'Fruit35_10_EfficientNetB7_300_1e-05_3'
}

general = {
    'plot_train_history': False,
    'plot_test_sample_eval': False,
    'experiment_mode': False,
    'save_model_info': False,
    'save_plot_eval': True
}

experiment_setup = {
    # 'datasets': ['Fruit35', 'CarParts', 'Fruit262'],
    'datasets': ['CarParts'],
    'models': ['EfficientNetB7'],
    'unfreeze_sizes': [50, 100],
    # 'unfreeze_sizes': [0, 50, 100, 200, 300, 400, 500],
    'epochs': [10, 15, 20, 30],
    'batch_sizes': [20, 32, 64],
    # 'batch_sizes': [64],
    # 'lrs': [0.00001, 0.000001]
    'lrs': [0.0000001]
}
