#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import os
from glob import glob
import random
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import OrderedDict

import archs

from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='outputs', help='ouput dir')
            
    args = parser.parse_args()

    return args

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    args = parse_args()

    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config['input_list'])

    model = model.cuda()

    dataset_name = config['dataset']
    img_ext = '.png'

    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'cvc':
        mask_ext = '.png'

    # Data loading code
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    # img_ids.sort()
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    ckpt = torch.load(f'{args.output_dir}/{args.name}/model.pth')

    try:        
        model.load_state_dict(ckpt)
    except:
        print("Pretrained model keys:", ckpt.keys())
        print("Current model keys:", model.state_dict().keys())

        pretrained_dict = {k: v for k, v in ckpt.items() if k in model.state_dict()}
        current_dict = model.state_dict()
        diff_keys = set(current_dict.keys()) - set(pretrained_dict.keys())

        print("Difference in model keys:")
        for key in diff_keys:
            print(f"Key: {key}")

        model.load_state_dict(ckpt, strict=False)
        
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)

            iou, dice, hd95_ = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            hd95_avg_meter.update(hd95_, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0

            os.makedirs(os.path.join(args.output_dir, config['name'], 'out_val'), exist_ok=True)
            for pred, img_id in zip(output, meta['img_id']):
                pred_np = pred[0].astype(np.uint8)
                pred_np = pred_np * 255
                img = Image.fromarray(pred_np, 'L')
                img.save(os.path.join(args.output_dir, config['name'], 'out_val/{}.jpg'.format(img_id)))

    
    print(config['name'])
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('HD95: %.4f' % hd95_avg_meter.avg)



if __name__ == '__main__':
    main()
