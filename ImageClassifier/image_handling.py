from PIL import Image
import PIL
from torchvision import transforms
import torch
import matplotlib.pyplot as plt


def imshow(image, ax=None, title=None, normalize=True, istensor = False):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    if istensor:
        image = image.numpy()
    image = image.transpose((1, 2, 0))
    
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax
data_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image_path)
    tensor_image = data_transforms(pil_image)
#     imshow(tensor_image)
    np_image = tensor_image.numpy()
    return np_image

def predict(image, model, topk=5):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image = image.unsqueeze_(0).to('cpu')
        model = model.to('cpu')
        
        model.eval()
        logps = model.forward(image)
        ps = torch.exp(logps).cpu()
        topk_probs, topk_classes = tuple(map(lambda e: e.numpy().squeeze(), ps.topk(topk, dim=1)))
    
    return topk_probs, topk_classes

