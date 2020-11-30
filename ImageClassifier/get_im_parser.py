# argument parser functional
import argparse

def get_input_args():
# Creates parse
    parser = argparse.ArgumentParser()
    # general options
    parser.add_argument('--verbosity', nargs='?', const=True, default=False,
                        help='gives more output information')
    # loading file options
    parser.add_argument('--dir', type=str, default='./flowers',
                        help='path to folder of images. Default ./flowers')
    parser.add_argument('--subdir', type=str, default='/test/1',
                        help='sub directory to image. Default /test/1')
    parser.add_argument('--image_filename', type = str, default = 'image_06743.jpg',
                        help='image file name to be processed')
    parser.add_argument('--label_filename', type=str, default = 'cat_to_name.json',
                        help='file with dictionary from classes to text labels')
    parser.add_argument('--hidden_units', type = str, default = '512' ,
                        help='number of hidden units (e.g. "523, 128") Def: 512')
    parser.add_argument('--topk', type = int, default = 5 ,
                        help='top most probable classes according to the model')
    parser.add_argument('--gpu', nargs='?', const=True, default=False ,
                        help="Call this option to use GPU processing")
 
    # model architecture and optimization options
    parser.add_argument('--arch', type=str, default = 'densenet',
                        help = 'Select NN architecture: densenet, vgg or alexnet')
    parser.add_argument('--load_state_file', type=str, default='checkpoint.pth',
                        help='specify state file to load')
    
    return parser.parse_args()