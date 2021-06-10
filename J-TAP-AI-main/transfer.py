import os
import argparse
import time
from PIL import Image

from wct import *

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms 
from torch.autograd import Variable
import torchvision.transforms.functional as tf


def wct_main():
    parser = argparse.ArgumentParser(description='Universal Style Transfer via Feature Transforms',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--content_image', default='./datasets/content/fatman.png', type=str, dest='content_image')
    parser.add_argument('--style_image', default='./datasets/style/wct2.jpg', type=str, dest='style_image')
    #parser.add_argument('--weight_path', default='./weights/vgg_conv.pth', type=str, dest='weight_path')
    parser.add_argument('--alpha', type=float,default=0.6, help='hyperparameter to blend wct feature and content feature')
    parser.add_argument('--img_size', default=512, type=int, dest='img_size')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder1 = './wct_weights/vgg_normalised_conv1_1.pth'
    encoder2 = './wct_weights/vgg_normalised_conv2_1.pth'
    encoder3 = './wct_weights/vgg_normalised_conv3_1.pth'
    encoder4 = './wct_weights/vgg_normalised_conv4_1.pth'
    encoder5 = './wct_weights/vgg_normalised_conv5_1.pth'   
    decoder1 = './wct_weights/feature_invertor_conv1_1.pth'
    decoder2 = './wct_weights/feature_invertor_conv2_1.pth'
    decoder3 = './wct_weights/feature_invertor_conv3_1.pth'
    decoder4 = './wct_weights/feature_invertor_conv4_1.pth'
    decoder5 = './wct_weights/feature_invertor_conv5_1.pth'

    paths = encoder1, encoder2, encoder3, encoder4, encoder5, decoder1, decoder2, decoder3, decoder4, decoder5
    wct = WCT(paths)

    
    content_image = Image.open(args.content_image).resize((args.img_size, args.img_size))
    print(content_image.mode)
    # if image mode is RGBA, CMYK etc.. => change RGB
    if content_image.mode != 'RGB':
        content_image = content_image.convert('RGB')
    content_image = tf.to_tensor(content_image)
    content_image.unsqueeze_(0)
    print('content_iamge shape: {}'.format(content_image.shape))

    style_image = Image.open(args.style_image).resize((args.img_size, args.img_size))
    # if image mode is RGBA, CMYK etc.. => change RGB
    if style_image.mode != 'RGB':
        style_image = style_image.convert('RGB')
    style_image = tf.to_tensor(style_image)
    style_image.unsqueeze_(0)
    print('style_image shape: {}'.format(style_image.shape))
    
    csf = torch.Tensor()

    cimg = Variable(content_image, volatile=True)
    simg = Variable(style_image, volatile=True)
    csf = Variable(csf)

    cimg = cimg.to(device)
    simg = simg.to(device)
    csf = csf.to(device)
    wct.to(device)

    # style transfer
    start_time = time.time()

    img = wct_style_transfer(wct, args.alpha, cimg, simg, csf).to(device)
    torchvision.utils.save_image(img, 'output.jpg')
    
    end_time = time.time()
    print('Start to End Time: %f' % (end_time - start_time))  


if __name__ == '__main__':
    #main()
    wct_main()