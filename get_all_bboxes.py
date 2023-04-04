"""Basically a heavily modified version of local_analysis_custom.py"""

##### MODEL AND DATA LOADING
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm

import re

import os

from helpers import  find_high_activation_crop
from preprocess import mean, std


gpu_id = '0'
modeldir = './saved_models/Resnet34/002'
model_ = '15push0.7530.pth'


img_id_to_path = pd.read_csv('datasets/CUB_200_2011/images.txt', sep=' ', header=None, names=['img_id', 'path'])
img_id_to_path = dict(img_id_to_path.values)

train_test_split = pd.read_csv('datasets/CUB_200_2011/train_test_split.txt', sep=' ', header=None, names=['img_id', 'is_train'])
train_test_split = dict(train_test_split.values)

img_id_to_path = {idx: path for idx, path in img_id_to_path.items() if train_test_split[idx] == 0}

# load the model

load_model_dir = modeldir #'./saved_models/vgg19/003/'
load_model_name = model_ #'10_18push0.7822.pth'

model_base_architecture = load_model_dir.split('/')[2]
experiment_run = '/'.join(load_model_dir.split('/')[3:])

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str)

ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
ppnet_multi = ppnet

img_size = ppnet_multi.img_size
prototype_shape = ppnet.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# load the test data and check test accuracy

##### SANITY CHECK
# confirm prototype class identity
load_img_dir = os.path.join(load_model_dir, 'img')

prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
prototype_img_identity = prototype_info[:, -1]


# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu().numpy()

# load the test image and forward it through the network
preprocess = transforms.Compose([
   transforms.Resize((img_size,img_size)),
   transforms.ToTensor(),
   normalize
])

all_bboxes = []

for id, img_path in tqdm(img_id_to_path.items()):
    # specify the test image to be analyzed

    test_image_path = 'datasets/CUB_200_2011/images/' + img_path
    test_image_label = int(img_path.split('.')[0]) #15

    img_pil = Image.open(test_image_path).convert('RGB')
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))

    images_test = img_variable.cuda()
    labels_test = torch.tensor([test_image_label])

    logits, min_distances = ppnet_multi(images_test)
    conv_output, distances = ppnet.push_forward(images_test)
    prototype_activations = ppnet.distance_2_similarity(min_distances)
    prototype_activation_patterns = ppnet.distance_2_similarity(distances)
    if ppnet.prototype_activation_function == 'linear':
        prototype_activations = prototype_activations + max_dist
        prototype_activation_patterns = prototype_activation_patterns + max_dist

    idx = 0

    array_act, sorted_indices_act = torch.sort(prototype_activations[idx])

    activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-1].item()].detach().cpu().numpy()
    upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                interpolation=cv2.INTER_CUBIC)

    # show the most highly activated patch of the image by this prototype
    high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)

    bbox_height_start = high_act_patch_indices[0]
    bbox_height_end = high_act_patch_indices[1]
    bbox_width_start = high_act_patch_indices[2]
    bbox_width_end = high_act_patch_indices[3]
    all_bboxes.append(np.array([id, bbox_width_start, bbox_height_start, bbox_width_end,  bbox_height_end]))

df = pd.DataFrame(all_bboxes, columns=['id', 'xmin', 'ymin', 'xmax', 'ymax'], index=None)
df.to_csv('bboxes.csv', index=False)
