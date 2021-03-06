import argparse
import numpy as np
import os
import torch
from torch import nn
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import logging

import opt
from evaluation import evaluate, evaluate_acc
from loss import InpaintingLoss
from net import PConvUNet
from net import VGG16FeatureExtractor
from models.resnet import resnet34
from places2 import Places2
from util.io import load_ckpt
from util.io import save_ckpt
from util.data_loader import DataSampler, GivenIterationSampler, create_logger

import json
import sys

class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='/srv/datasets/Places2')
parser.add_argument('--mask_root', type=str, default='./mask')
parser.add_argument('--save_dir', type=str, default='./snapshots/default')
parser.add_argument('--log_dir', type=str, default='./logs/default')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_finetune', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=1000000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--vis_interval', type=int, default=5000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--resume', type=str)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--classify', action='store_true')
parser.add_argument('--attention', action='store_true')

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images'.format(args.save_dir))
    os.makedirs('{:s}/ckpt'.format(args.save_dir))

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

size = (args.image_size, args.image_size)
img_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])


src = os.path.join(args.root, 'dict.json')
with open(src) as fr:
    label_dict = json.loads(fr.read())


# model = PConvUNet(class_num=len(label_dict)).to(device)
model = resnet34(num_classes=len(label_dict), pretrained=True, attention=args.attention).to(device)

if args.classify:
    model.classify()

if args.finetune:
    lr = args.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = args.lr

start_iter = 0
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device) if not args.classify \
        else nn.CrossEntropyLoss()

if args.resume:
    start_iter = load_ckpt(
        args.resume, [('model', model)], [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

dataset_train = Places2(args.root, args.mask_root, img_tf, mask_tf, 'train', label_dict, args.classify)
dataset_val = Places2(args.root, args.mask_root, img_tf, mask_tf, 'val', label_dict, args.classify)

train_loader = data.DataLoader(
    dataset_train, batch_size=args.batch_size,
    sampler=GivenIterationSampler(dataset_train, args.max_iter, args.batch_size, start_iter),
    num_workers=args.n_threads)
print(len(dataset_train))

val_sampler = DataSampler(dataset_val)
val_loader = data.DataLoader(
            dataset_val, batch_size=args.batch_size, shuffle=False,
            num_workers=args.n_threads, pin_memory=False, sampler=val_sampler)



logger = create_logger('global_logger', args.log_dir+'/log.txt')

# logger = logging.getLogger('global_logger')

# for i in tqdm(range(start_iter, args.max_iter)):
for i, (image, mask, gt, label) in enumerate(train_loader):
    cur_step = i + start_iter
    model.train()

    image, mask, gt, label = image.to(device), mask.to(device), gt.to(device), label.to(device)

    if args.classify:
        # pred = model.classify(image)
        pred = model(image)
        loss = criterion(pred, label)

        if (cur_step + 1) % args.log_interval == 0:
            writer.add_scalar('loss_{:s}'.format('pred'), loss.item(), cur_step + 1)

            logger.info('Iter: %d\tLoss %.4f' % (cur_step, loss))

    else:
        output, _ = model(image, mask)
        loss_dict = criterion(image, mask, output, gt)

        loss = 0.0
        for key, coef in opt.LAMBDA_DICT.items():
            value = coef * loss_dict[key]
            loss += value
            if (cur_step + 1) % args.log_interval == 0:
                writer.add_scalar('loss_{:s}'.format(key), value.item(), cur_step + 1)
                logger.info('Iter: %d\tLoss_%s %.4f' % (cur_step, key, value.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (cur_step + 1) % args.save_model_interval == 0 or (cur_step + 1) == args.max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, cur_step + 1),
                  [('model', model)], [('optimizer', optimizer)], cur_step + 1)

    if (cur_step + 1) % args.vis_interval == 0:
        model.eval()
        if args.classify:
            acc = evaluate_acc(model, val_loader, device)
            writer.add_scalar('test_acc', acc, cur_step + 1)
            logger.info('Iter: %d\tAcc %.4f' % (cur_step, acc))
        else:
            evaluate(model, dataset_val, device,
                     '{:s}/images/test_{:d}.jpg'.format(args.save_dir, cur_step + 1))

writer.close()
