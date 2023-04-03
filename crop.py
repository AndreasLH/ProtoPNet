from concurrent.futures import ThreadPoolExecutor as Executor

import pandas as pd
from PIL import Image
from tqdm import tqdm

from helpers import makedir

img_id_to_path = pd.read_csv('datasets/CUB_200_2011/images.txt', sep=' ', header=None, names=['img_id', 'path'])
img_id_to_path = dict(img_id_to_path.values)

train_test_split = pd.read_csv('datasets/CUB_200_2011/train_test_split.txt', sep=' ', header=None, names=['img_id', 'is_train'])
train_test_split = dict(train_test_split.values)

img_id_to_bbx = pd.read_csv('datasets/CUB_200_2011/bounding_boxes.txt', sep=' ', header=None, names=['img_id', 'x', 'y', 'w', 'h'])

makedir('datasets/cub200_cropped/')
makedir('datasets/cub200_cropped/train_cropped/')
makedir('datasets/cub200_cropped/test_cropped/')

for key, img_path in tqdm(img_id_to_path.items()):
    img_folder = img_path.split('/')[0]
    makedir('datasets/cub200_cropped/train_cropped/' + img_folder)
    makedir('datasets/cub200_cropped/test_cropped/'  + img_folder)

def process_image(key, img_path):
    img = Image.open('datasets/CUB_200_2011/images/' + img_path)
    x, y, w, h = img_id_to_bbx[img_id_to_bbx['img_id'] == key].values[0][1:]
    img = img.crop((x, y, x + w, y + h))
    img_path_minus_ext = img_path.split('.')[0] + '.' + img_path.split('.')[1]
    # 1 is training, 0 is testing
    if train_test_split[key] == 1:
        img.save('datasets/cub200_cropped/train_cropped/' + str(img_path_minus_ext) + '.png')
    else:
        img.save('datasets/cub200_cropped/test_cropped/' + str(img_path_minus_ext) + '.png')

with tqdm(total=len(img_id_to_path)) as progress:
    with Executor(max_workers=8) as executor:
        for __ in executor.map(process_image, img_id_to_path.keys(), img_id_to_path.values()):
            progress.update()
