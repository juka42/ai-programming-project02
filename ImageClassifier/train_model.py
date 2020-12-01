
from torch import nn
from torchvision import models
from collections import OrderedDict
from torch import optim
import torch

#define classifier architecture
def classifier_arch(n_inputs, n_output, hidden_units):
    i = 0
    hidden_layers = list(map(int, hidden_units.split(', ')))
    num_param_previous = n_inputs
    classifier_arch = OrderedDict([])
    for hl in hidden_layers:
        i+=1
        classifier_arch['fc'+str(i)]=nn.Linear(num_param_previous, hl)
        classifier_arch['relu'+str(i)]=nn.ReLU()
        classifier_arch['dropout'+str(i)]=nn.Dropout(0.5)
        num_param_previous = hl
    i+=1
    classifier_arch['fc'+str(i)]=nn.Linear(num_param_previous, n_output)
    classifier_arch['output']=nn.LogSoftmax(dim=1)
    return classifier_arch

def define_model(args, n_outputs):
    #################
    ## build network

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

    num_inputs = {'alexnet': 9216,'vgg':25088,'densenet':1024}
       ### classifier_arch defines an ordinary dict with the model architecture
    classifier_architecture = classifier_arch(num_inputs[args.arch], n_outputs, args.hidden_units)
       ### here we define the model structure for the classifier part
    classifier = nn.Sequential(classifier_architecture)
       ### here we substitute the model classifier with our 102 categories classifier
    model.classifier = classifier
    return model, classifier_architecture

# train model function
def train_model(args,
                model,
                trainloader,
                validloader,
                device,
                learning_rate = 0.002,
                _criterion= nn.NLLLoss(),
                print_every = 32,
                verbosity = True
               ):
    criterion = _criterion
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    steps = 0
    running_loss = 0

    model.train()
    for epoch in range(args.epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss, valid_accuracy = calculate_accuracy(model, validloader, device)
                model.train()
                print(f"E {epoch+1}/{args.epochs}.. "
                      f"Trainning:  Loss={running_loss/print_every:.3f}.. "
                      f"Validation:  Loss={valid_loss:.3f}.. "
                      f"Accuracy={100*valid_accuracy:.3f}")
                running_loss = 0
                model.train()
    pass



#calculate accuracy functional
def calculate_accuracy(model, loader, device, _criterion = nn.NLLLoss()):
    criterion = _criterion
    loss = 0
    accuracy = 0
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            loss += criterion(logps, labels)

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    loss = loss/len(loader)
    accuracy=accuracy/len(loader)
    return loss, accuracy
