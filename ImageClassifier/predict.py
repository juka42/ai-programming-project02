from train_model import define_model
from get_im_parser import get_input_args
from label_mapping import label_mapping
from read_images import read_images
from save_and_load_nn import load_checkpoint
from image_handling import process_image, imshow, predict

args = get_input_args()

###############
#label mapping
# this defines a cat to name dictionary

trainloader, validloader, testloader, class_to_idx \
    = read_images(args.dir)


cat_to_name, idx_to_class, idx_to_label \
    = label_mapping(args.label_filename, class_to_idx)
n_outputs = len(cat_to_name)

model, classif_arch = define_model(args, n_outputs)


#########################
# load model
if args.load_state_file != None:
    model.classifier = load_checkpoint(args)

image_file = args.dir+args.subdir+'/'+args.image_filename

image = process_image(image_file)
# imshow(image)

probs, classes = predict(image, model, args.topk)
print(f'top {args.topk} classes:       {classes}')
print(f'top {args.topk} probabilities: {probs}')
print(f'most probable class is: {idx_to_class[classes[0]]} or {idx_to_label[classes[0]]}')

