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
                        help='path to folder of images')
    parser.add_argument('--label_filename', default = 'cat_to_name.json',
                        help='file with dictionary from classes to text labels')

    # model architecture and optimization options
    parser.add_argument('--arch', type=str, default = 'densenet',
                        help = 'Select NN architecture: densenet, vgg or alexnet')
    parser.add_argument('--learning_rate', type=float, default = 0.002,
                        help='select learning ratte. Default = 0.01')
    parser.add_argument('--hidden_units', type = str, default = '512' ,
                        help='number of hidden units (e.g. "523, 128") Def: 512')
    parser.add_argument('--epochs', type = int, default = 1 ,
                        help='Select number of epochs. Default = 20')
    parser.add_argument('--printevery', type = int, default = 32 ,
                        help='Latency of print during trainning')
    parser.add_argument('--gpu', nargs='?', const=True, default=False,
                        help="Call this option to use GPU processing")
    parser.add_argument('--save_state_file', type = str, default = 'checkpoint.pth',
                        help='prevent saving model state')
    parser.add_argument('--no_train', nargs='?', const=True, default=False,
                        help='use not to train parameters')
    parser.add_argument('--load_state_file', type=str, default='checkpoint.pth',
                        help='specify state file to load')

    return parser.parse_args()
