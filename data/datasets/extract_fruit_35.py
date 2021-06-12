import os
import random
import shutil


SRC_PATH = os.path.join(os.getcwd(), 'fruit_262_cpy/pure')
PATH = os.path.join(os.getcwd(), 'fruit_35')
pure_dir = os.path.join(PATH, 'pure')

CLASSES_TO_CPY = {
    'apple',
    'avocado',
    'banana',
    'bell pepper',
    'cherry',
    'coffee',
    'coconut',
    'fig',
    'grape',
    'jalapeno',
    'lime',
    'mango',
    'naranjilla',
    'olive',
    'orange',
    'papaya',
    'watermelon',
    'zucchini',
    'strawberry',
    'pear',
    'persimmon',
    'pineapple',
    'monkfruit',
    'mundu',
    'lychee',
    'kundong',
    'goji',
    'emblic',
    'grenadilla',
    'jambui',
    'japanese raisin',
    'jasmine',
    'kiwi',
    'mandarine',
    'melinjo',
    'nance'
}
SAMPLE_SIZE = 100

for subdir in [f.path for f in os.scandir(SRC_PATH) if f.is_dir()]:
    if os.path.basename(subdir) in CLASSES_TO_CPY:
        pure_src = os.path.join(SRC_PATH, os.path.basename(subdir))
        pure_dest = os.path.join(pure_dir, os.path.basename(subdir))

        if not os.path.isdir(pure_dest):
            os.makedirs(pure_dest)

        for i in range(SAMPLE_SIZE):
            file = random.choice(os.listdir(subdir))
            file_path = os.path.join(pure_src, file)
            shutil.move(file_path, pure_dest)
