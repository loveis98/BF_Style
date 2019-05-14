import argparse
import logging
import os
import random
import re
import sys
import time
import collections
import glob
import cv2
import numpy as np
import torch
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torchvision.transforms.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import scipy.misc
from PIL import Image
import models.resnet as lw
import models.fast_scnn as fs


NORMALISE_PARAMS = [1./255, # SCALE
                    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)), # MEAN
                    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))] # STD

MODEL = 'lw_refine'
SNAPSHOT_DIR = './ckpt/'
CKPT_PATH = './ckpt/lw_refine.pth.tar'


class Normalise(object):

    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        return {'image': (self.scale * image - self.mean) / self.std, 'mask' : sample['mask']}

class ToTensor(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}
    
class Resize(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = Image.fromarray(image.astype('uint8'))
        t = transforms.Compose([transforms.Resize((800, 800))])        
        return {'image': np.array(t(image)),
                'mask': mask}

class Dataset(Dataset):

    def __init__(self, data_file, data_dir, transform_trn=None, transform_val=None):
        
        with open(data_file, 'r') as f:
            datalist = f.readlines()
        self.datalist = [(k, v) for k, v in map(lambda x: x.strip('\n').split(' '), datalist)]
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = 'train'

    def set_stage(self, stage):
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.datalist[idx][0])
        msk_name = os.path.join(self.root_dir, self.datalist[idx][1])
        def read_image(x):
            img_arr = np.array(Image.open(x))
            if len(img_arr.shape) == 2: # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr
        image = read_image(img_name)
        mask = np.array(Image.open(msk_name))
        if img_name != msk_name:
            assert len(mask.shape) == 2, 'Masks must be encoded without colourmap'
        sample = {'image': image, 'mask': mask}
        if self.stage == 'train':
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == 'val':
            if self.transform_val:
                sample = self.transform_val(sample)
        return sample



def get_arguments():
    
    parser = argparse.ArgumentParser(description="Full Pipeline Training")

    parser.add_argument('--model', type=str, default=MODEL,
                        help='model name (default: lw_refine).')
    parser.add_argument("--test-dir", type=str, default='./dataset/',
                        help="Path to the test set directory.")
    parser.add_argument("--test-list", type=str, nargs='+', default='./dataset/test.txt',
                        help="Path to the test set list.")
    parser.add_argument("--out-dir", type=str, nargs='+', default='test_result',
                        help="Path to the test set result.")
    parser.add_argument("--ckpt-path", type=str, default=CKPT_PATH,
                        help="Path to the checkpoint file.")

    return parser.parse_args()


def load_ckpt(
    ckpt_path, ckpt_dict
    ):
    best_val = epoch_start = 0
    if os.path.exists(args.ckpt_path):
        ckpt = torch.load(ckpt_path)
        for (k, v) in ckpt_dict.items():
            if k in ckpt:
                v.load_state_dict(ckpt[k])
    return best_val, epoch_start


def validate(segmenter, val_loader, num_classes=-1, outdir='test_result'):
    
    pallete = [
        0, 0, 0,
        255, 255, 255,
    ]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    segmenter.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            input = sample['image']
            target = sample['mask']
            input = torch.autograd.Variable(torch.reshape(input, (1, 3, input.shape[2], input.shape[3]))).float().cuda()
                
            output = segmenter(input)

            if MODEL == 'lw_refine':
                output = cv2.resize(output[0, :num_classes].data.cpu().numpy().transpose(1, 2, 0),
                                    target.size()[1:][::-1],
                                    interpolation=cv2.INTER_CUBIC).argmax(axis=2).astype(np.uint8)
                
            else:
                output = output[0].squeeze(0)
                output = cv2.resize(output.data.cpu().numpy().transpose(1, 2, 0),
                                    target.size()[1:][::-1],
                                    interpolation=cv2.INTER_CUBIC).argmax(axis=2).astype(np.uint8)
            
            out_img = Image.fromarray(output.astype('uint8'))
            out_img.putpalette(pallete)
            out_img.save(outdir + '/seg{}.png'.format(i))
            
            im = cv2.imread(outdir + '/seg{}.png'.format(i), cv2.IMREAD_GRAYSCALE)
            kernel = np.ones((9, 9), np.uint8)
            opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, iterations=2)
            cv2.imwrite(outdir + '/seg{}.png'.format(i), opening)

def main():
    global args
    args = get_arguments()
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(42)
    global MODEL
    MODEL = args.model
    if MODEL == 'lw_refine':
        segmenter = nn.DataParallel(lw.rf_lw50(2, True)).cuda()
        best_val, epoch_start = load_ckpt(args.ckpt_path, {'segmenter' : segmenter})
    else:
        segmenter = fs.get_fast_scnn().to(torch.device("cuda:0"))
    torch.cuda.empty_cache()
    if MODEL == 'fast_scnn':
        composed_val = transforms.Compose([Resize(), Normalise(*NORMALISE_PARAMS), ToTensor()])
    else:
        composed_val = transforms.Compose([Normalise(*NORMALISE_PARAMS), ToTensor()])
    testset = Dataset(data_file=args.test_list,
                     data_dir=args.test_dir,
                     transform_trn=None,
                     transform_val=composed_val)

    test_loader = DataLoader(testset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=16,
                             pin_memory=True)
    test_loader.dataset.set_stage('val')

    return validate(segmenter, test_loader, num_classes=2, outdir=args.out_dir)


if __name__ == '__main__':
    main()
