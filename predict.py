#用于处理命令行参数
import argparse
#用于记录日志信息
import logging
#提供与操作系统交互的功能，如文件路径操作
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

#输入图像生成掩膜
def predict_img(net, #训练好的模型
                full_img, #输入的完整的图像
                device, #计算设备（如CPU或GPU）
                scale_factor=1, #缩放因子，调整输入图像的大小
                out_threshold=0.5): #输出的阀值
    net.eval() #设置模型为评估模式，会禁用某些层（如Dropout和BatchNorm）的训练行为
    #预处理
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    #将图像的形状从（C，H，W）转换为（1，C，H，W）
    img = img.unsqueeze(0)
    #将图像转移到GPU或CPU，并确保数据类型是float32
    img = img.to(device=device, dtype=torch.float32)
    #禁用梯度计算
    with torch.no_grad():
        #前向传播，将图像输入模型，得到输出，并将结果移到CPU
        output = net(img).cpu()
        #上采样，调整输出图像的大小与原图像一致，运用双线性插值
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        #如果类别大于1，得到最大的类别索引
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else: #类别等于1，sigmoid函数计算概率，与阀值比较生成二进制掩膜
            mask = torch.sigmoid(output) > out_threshold
   #返回掩膜的数组
    return mask[0].long().squeeze().numpy()


def get_args():
    #创建命令行参数解释器
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    #指定模型文件的路径
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    #指定输入掩膜的文件名
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    #指定输出掩膜的文件名
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    #可视化
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    #不保存输出参数
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    #掩膜阈值的参数
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    #缩放因子
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    #双线性插值
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    #类别数量
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()

#根据输入文件名生成输出文件名，后缀为_OUT.png
def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    #args是通过argparse模块解析命令行参数得到的对象，代表从命令行接收的参数
    #如果没有指定输出文件名，则根据输入文件名生成输出文件名
    return args.output or list(map(_generate_name, args.input))

#将掩膜转换为图像
def mask_to_image(mask: np.ndarray, mask_values):

    #如果掩膜值是列表，初始化一个多通道数组
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)

    #如果掩膜值为[0,1]，初始化一个布尔数组
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)

    #初始化一个单通道的输出数组
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    #如果掩膜是多类，转换为单通道掩膜
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    #将输出数组的值设为mask_values
    for i, v in enumerate(mask_values):
        out[mask == i] = v

    #将输出数组转换为PIL图像
    return Image.fromarray(out)


if __name__ == '__main__':
    #获取命令行参数并设置日志格式
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    #从命令行参数中获取输入文件名，并生成相应的输出文件名
    in_files = args.input
    out_files = get_output_filenames(args)

    #创建一个U-Net实例，设置输入通道数、类别数和是否使用双线性插值
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    #判断是否使用GPU，并记录加载模型的信息。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    #将模型参数加载到U-Net中，并提取掩膜值
    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    #记录模型已加载的信息。
    logging.info('Model loaded!')

    #对每个输入文件进行处理，记录当前处理的图像。
    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        #生成图像的掩膜。
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        #如果未指定不保存，则将生成的掩膜保存为文件，并记录保存信息。
        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        #如果指定了可视化参数，调用绘图函数显示原图和掩膜。
        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
