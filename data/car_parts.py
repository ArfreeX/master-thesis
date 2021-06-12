import os
import shutil

from data.constants import CAR_PARTS_DIR
from data.tl_dataset import TLDataset


class CarParts(TLDataset):

    def __init__(self):
        super().__init__()
        self._dataset_dir = CAR_PARTS_DIR
        self._train_dir = os.path.join(CAR_PARTS_DIR, 'train')
        self._validation_dir = os.path.join(CAR_PARTS_DIR, 'validation')
        self._test_dir = os.path.join(CAR_PARTS_DIR, 'test')

    def prepare_dataset(self):
        if not os.path.isdir(os.path.join(CAR_PARTS_DIR, 'pure')):
            raise Exception("Missing \'pure\' directory.")

        if os.path.isdir(self._train_dir):
            shutil.rmtree(self._train_dir, ignore_errors=True)
        if os.path.isdir(self._validation_dir):
            shutil.rmtree(self._validation_dir, ignore_errors=True)
        if os.path.isdir(self._test_dir):
            shutil.rmtree(self._test_dir, ignore_errors=True)

        def remove_subdir(parent, subdirectory):
            if os.path.isdir(subdirectory):
                for files in os.listdir(subdirectory):
                    shutil.move(os.path.join(subdirectory, files), parent)
                shutil.rmtree(subdirectory)

        pure_dir = os.path.join(CAR_PARTS_DIR, 'pure')
        remove_subdir(pure_dir, os.path.join(pure_dir, 'Car parts'))
        remove_subdir(pure_dir, os.path.join(pure_dir, 'External'))
        remove_subdir(pure_dir, os.path.join(pure_dir, 'Internal'))
