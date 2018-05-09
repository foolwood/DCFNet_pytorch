from dataset import VID
from net import DCFNet
import torch
from torch.utils.data import dataloader
import torch.nn as nn
import torch.backends.cudnn as cudnn

import argparse

parser = argparse.ArgumentParser(description='Training DCFNet in Pytorch 0.4.0')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

args = parser.parse_args()

model = DCFNet()
model.cuda()
model = torch.nn.parallel.DistributedDataParallel(model)

criterion = nn.MSELose().cuda()

optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

cudnn.benchmark = True

train_dataset = VID(file='dataset.json', trian=True)
val_dataset = VID(file='dataset.json', trian=False)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# if args.evaluate:
#     validate(val_loader, model, criterion)
#     return
#
# for epoch in range(args.start_epoch, args.epochs):
#     adjust_learning_rate(optimizer, epoch)
#
#     # train for one epoch
#     train(train_loader, model, criterion, optimizer, epoch)
#
#     # evaluate on validation set
#     prec1 = validate(val_loader, model, criterion)
#
#     # remember best prec@1 and save checkpoint
#     is_best = prec1 > best_prec1
#     best_prec1 = max(prec1, best_prec1)
#     save_checkpoint({
#         'epoch': epoch + 1,
#         'arch': args.arch,
#         'state_dict': model.state_dict(),
#         'best_prec1': best_prec1,
#         'optimizer' : optimizer.state_dict(),
#     }, is_best)