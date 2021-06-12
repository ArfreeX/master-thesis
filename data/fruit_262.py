import os
import shutil

from data.constants import FRUIT_262_DIR
from data.tl_dataset import TLDataset


class Fruit262(TLDataset):

    def __init__(self):
        super().__init__()
        self._dataset_dir = FRUIT_262_DIR
        self._train_dir = os.path.join(FRUIT_262_DIR, 'train')
        self._validation_dir = os.path.join(FRUIT_262_DIR, 'validation')
        self._test_dir = os.path.join(FRUIT_262_DIR, 'test')

    def prepare_dataset(self):
        if not os.path.isdir(os.path.join(FRUIT_262_DIR, 'pure')):
            raise Exception("Missing \'pure\' directory.")

        if os.path.isdir(self._train_dir):
            shutil.rmtree(self._train_dir, ignore_errors=True)
        if os.path.isdir(self._validation_dir):
            shutil.rmtree(self._validation_dir, ignore_errors=True)
        if os.path.isdir(self._test_dir):
            shutil.rmtree(self._test_dir, ignore_errors=True)
