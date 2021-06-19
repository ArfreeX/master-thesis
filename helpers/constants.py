import os


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots_car_parts_vgg_unf')
MAIN_CSV_PATH = os.path.join(RESULTS_DIR, 'experiment_log_car_parts_vgg_unf.csv')
