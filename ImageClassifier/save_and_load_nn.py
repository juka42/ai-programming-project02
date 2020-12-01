##### Functions to save, load and define nn architecture
from collections import OrderedDict
from torch import nn
import torch
from torchvision import models

#this function receives number of inputs, number of outputs and number of h
#    idden layers and returns an ordered dictionary to use witn nn.Sequential

# This function loads the nn architecture in an ordered dictionary
#      and parameter states, returnung the nn for the classifier
def load_checkpoint(args):
    # Read the pretrained network
    if args.arch == "densenet":
        model = models.densenet121(pretrained=True)
    elif args.arch == "vgg":
        model = models.vgg11(pretrained=True)
    elif args.arch == "alexnet":
        model = models.alexnet(pretrained=True)
    else: print("Please define a valid architecture.")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False    # define classifier architecture

    checkpoint = torch.load(args.load_state_file)
    classifier_arch = checkpoint['architecture']
    classifier = nn.Sequential(classifier_arch)
    classifier.load_state_dict(checkpoint['state_dict'])
    print(f'model loaded from file:   {args.load_state_file}')
    return classifier
#model.classifier = load_checkpoint(state_file_path)
#model.classifier

# This function saves the nn architecture in an ordered dictionary
#      and parameter states in a specified file
def save_estate(model, classifier_arch, args):
    checkpoint={}
    checkpoint['architecture'] = classifier_arch
    checkpoint['state_dict'] = model.state_dict()
    torch.save(checkpoint, args.save_state_file)
    print('model saved')
    pass
