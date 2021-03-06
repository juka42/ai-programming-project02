# imports
import read_images

from get_parser import get_input_args
import read_images
from label_mapping import label_mapping
from train_model import define_model, classifier_arch, calculate_accuracy, train_model
from save_and_load_nn import load_checkpoint, save_estate
from torchvision import models
import torch
from torch import optim
import torch.nn.functional as F
from torch import nn
from datetime import date


import numpy as np

from PIL import Image
import PIL

import matplotlib.pyplot as plt
from collections import OrderedDict

# import os

#get arguments from command line
args = get_input_args()

#get images into loaders
trainloader, validloader, testloader, class_to_idx \
    = read_images.read_images(args.dir)

###############
#label mapping
# this defines a cat to name dictionary
cat_to_name, idx_to_class, idx_to_label \
    = label_mapping(args.label_filename, class_to_idx)
n_outputs = len(cat_to_name)

model, classif_arch = define_model(args, n_outputs)

# set device
device = 'cpu'
if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print("Warning: GPU specified but not avaliable. Proceding with CPU")
model.to(device)

#########################
# load model
if args.load_state_file != None:
    print('\nloading model')
    model.classifier = load_checkpoint(args)
    loss, accuracy = calculate_accuracy(model, testloader, device)
    print('\nTest performance just loaded:')
    print(f'Test loss = {loss} Test accuracy {accuracy}\n')

##########################
# train model
if not args.no_train:
    print('#### Train model ####')
    train_model(args, model, trainloader, validloader, device, verbosity = args.verbosity)

    ########################
    ### Save the checkpoint
    if args.save_state_file == 'auto':
        ac = str(int(test_accuracy*100))
        dt = str(date.today())

        state_file_path = 'checkpoint_' + pretrained + '_ac' + ac + '_'+ dt +'.pth'
        args.save_state_file = state_file_path
        print(f'\n######\n {args.save_state_file} \n##########\n')
        save_estate(model.classifier, classif_arch, args)
    elif args.save_state_file != 'none':
        print(f'\nsaving file: {args.save_state_file}\n')
        save_estate(model.classifier, classif_arch, args)
    else:
        print('#####\nstate not saved\n##########\n')

else:
    print("Note: Not training model parameters")

# display accuracy loss etc

loss, accuracy = calculate_accuracy(model, testloader, device)

print('Test los = {:.3f} --- Test Accuracy = {:.3f}'.format(loss,accuracy*100))
