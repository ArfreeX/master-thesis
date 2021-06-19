from data.fruit_262 import Fruit262
from data.fruit_35 import Fruit35
from data.car_parts import CarParts


def prepare_data(config):
    test_predefined = False

    if config['dataset'] == 'Fruit262':
        data = Fruit262()
    elif config['dataset'] == 'Fruit35':
        data = Fruit35()
        test_predefined = True
    elif config['dataset'] == 'CarParts':
        data = CarParts()
    else:
        raise Exception("Unknown dataset selected")

    if config['prepare_data']:
        data.prepare_dataset()
    if config['split_dataset']:
        data.split_dataset(config['split_percentage'], test_predefined)

    data.load_data(config['img_size'], config['batch_size'])
    if config['prefetch']:
        data.prefetch()

    return {
        'train': data.get_train(),
        'validation': data.get_validation(),
        'test': data.get_test(),
        'categories': data.get_categories()
    }
