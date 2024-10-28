import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask.squeeze().long().numpy()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')  # 修改为3类
    
    return parser.parse_args()

def get_output_filenames(args, class_id=None):
    def _generate_name(fn, class_id=None):
        base_name = os.path.splitext(fn)[0]
        if class_id is not None:
            return f'{base_name}_class_{class_id}_OUT.png'
        return f'{base_name}_OUT.png'

    if args.output:
        return args.output
    else:
        if class_id is None:
            return list(map(_generate_name, args.input))
        else:
            return list(map(lambda x: _generate_name(x, class_id), args.input))

def mask_to_image(mask: np.ndarray):
    """Convert a mask with class indices to an image directly using those indices"""
    return Image.fromarray(mask.astype(np.uint8))

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input

    # Load model
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        # Predict mask
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # 输出掩码中的唯一标签
        unique_labels = np.unique(mask)
        logging.info(f'Unique labels in mask for image {filename}: {unique_labels}')

        # Save binary masks for each class
        if not args.no_save:
            for class_id in range(args.classes):
                out_filename = get_output_filenames(args, class_id)[i]
                class_mask = (mask == class_id).astype(np.uint8) * 255  
                result = mask_to_image(class_mask)
                result.save(out_filename)
                logging.info(f'Mask for class {class_id} saved to {out_filename}')

        # Visualize results if requested
        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
