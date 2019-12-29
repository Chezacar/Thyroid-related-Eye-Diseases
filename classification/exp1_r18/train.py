# Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import resnet
from sklearn.metrics import roc_auc_score
from scipy import special

import TAO_loader

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='TAO training')
parser.add_argument('--epochs', default=160, type= int, metavar='N', help='number of total epochs')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--workers',default=6,type=int, metavar='N')
# parser.add_argument('--num_classes', default=2, type=int, metavar='N')
parser.add_argument('-b', '--batch_size',default=64, type=int,metavar='N')
parser.add_argument('--lr',default=0.01,type = float, metavar='LR')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--sum_freq', default=100, type=int, metavar='N', help='summary frequency (default: 100)')
parser.add_argument('--save_freq', default=5, type=int, metavar='N', help='save checkpoint frequency (default: 5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-sp', '--summary_path', default='', type=str, metavar='PATH', help='path to store summary event file')
parser.add_argument('-cp','--checkpoint_path', default='', type=str, metavar='PATH', help='path to store checkpoint')
parser.add_argument('-op', '--output_path', default='', type=str, metavar='PATH', help='path to store test output')
parser.add_argument('--suffix', default ='', type = str, help = 'suffix of summmary and checkpoint dir')
parser.add_argument('--lr_path', default='', type=str, metavar='PATH', help='path to lr file')

parser.add_argument('--threshold',default=0.5,type = float, metavar='THRESHOLD')

parser.add_argument('--trainval_file', default='/DB/rhome/zdcheng/workspace/hyperthyreosis_eye/classification/all_trainval.json',
                    type=str, metavar='PATH', help='path to train json file')
parser.add_argument('--valfold', default=0,
                    type=int, metavar='N', help='fold Indicator')
parser.add_argument('--data_root',default='/DATA5_DB8/data/zdcheng/hyperthyreosis_eye/class_arraydata', type = str,
                    metavar='PATH',help='path to image data root')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

parser.add_argument('--fake', dest='fake', action='store_true', help='use fake data')

best_acc = 0
best_prec = 0
best_recall = 0
best_F1 = 0
best_AUC = 0

def main():
    global args, best_acc, best_prec, best_recall, best_F1, best_AUC, writer
    args = parser.parse_args()

    args.summary_path = os.path.join(args.summary_path, args.suffix)
    if not os.path.exists(args.summary_path):
        os.makedirs(args.summary_path)
    writer = SummaryWriter(args.summary_path)

    args.checkpoint_path = os.path.join(args.checkpoint_path, args.suffix)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    args.output_path = os.path.join(args.output_path, args.suffix)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    model = resnet.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 1)
    model = torch.nn.DataParallel(model).cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            best_prec = checkpoint['best_prec']
            best_recall = checkpoint['best_recall']
            best_F1 = checkpoint['best_F1']
            best_AUC = checkpoint['best_AUC']
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise RuntimeError()

    train_dataset =  TAO_loader.TAO(args.trainval_file, args.data_root, mode='train', val_fold=args.valfold, fake=False )
    val_dataset = TAO_loader.TAO(args.trainval_file, args.data_root, mode='val', val_fold=args.valfold, fake=False )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers, pin_memory=True)
    global epoch_len
    epoch_len = len(train_loader)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=int(args.batch_size * 0.5), shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    Acc, Prec, Recall, F1, AUC = validate(val_loader, model, criterion, -1)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_path)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        Acc,Prec,Recall,F1,AUC = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = Acc > best_acc
        best_acc = max(Acc, best_acc)
        best_prec = max(Prec, best_prec)
        best_recall = max(Recall, best_recall)
        best_F1 = max(F1, best_F1)
        best_AUC = max(AUC, best_AUC)

        save_checkpoint({
            'epoch': epoch,
            #'arch': args.arch,
            'model_state': model.state_dict(),
            'best_acc': best_acc,
            'best_prec': best_prec,
            'best_recall': best_recall,
            'best_F1': best_F1,
            'best_AUC': best_AUC,
            'optimizer' : optimizer.state_dict(),
        }, is_best, epoch)

def train(trainloader, model, criterion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accMeter = AverageMeter()

    model.train()

    end = time.time()

    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.to(torch.float32).cuda()

        # compute output
        output = model(input)
        loss = criterion(output.view(output.size(0)), target)

        # measure accuracy and record loss
        acc = accuracy(output.detach(), target, args.threshold)
        losses.update(loss.item(), input.size(0))
        accMeter.update(acc, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {accMeter.val:.3f} ({accMeter.avg:.3f})'.format(
                epoch, i, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, accMeter=accMeter))

        global_step = epoch * epoch_len + i
        if i % args.sum_freq == 0:
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.add_scalar('train_acc', acc, global_step)
    global_step = epoch * epoch_len + epoch_len - 1
    writer.add_scalar('epochavg_train_loss', losses.avg, global_step)
    writer.add_scalar('epochavg_train_acc', accMeter.avg, global_step)

def validate(val_loader, model, criterion, epoch):
    # epoch < 0 means no summary
    batch_time = AverageMeter()
    losses = AverageMeter()
    accMeter = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.to(torch.float32).cuda()

        # compute output
        output = model(input)
        loss = criterion(output.view(output.size(0)), target)

        # measure accuracy and record loss
        acc = accuracy(output.detach(), target, args.threshold)
        losses.update(loss.item(), input.size(0))
        accMeter.update(acc, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Val: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {accMeter.val:.3f} ({accMeter.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                accMeter=accMeter))
        if i == 0:
            out_array = output.detach().cpu().numpy()
            target_array = target.detach().cpu().numpy()
        else:
            out_array = np.concatenate((out_array, output.detach().cpu().numpy()), axis=0)
            target_array = np.concatenate( (target_array, target.detach().cpu().numpy()), axis = 0)
    out_array = out_array.reshape(out_array.shape[0])
    Prec,Recall,F1 = precision_recall_F1(out_array,target_array,threshold=args.threshold)
    AUC = AUROC(out_array, target_array)

    print(' * Acc {accMeter.avg:.3f} '.format(accMeter=accMeter))
    print(' * Prec {:.4f}'.format(Prec))
    print(' * Recall {:.4f}'.format(Recall))
    print(' * F1 {:.4f}'.format(F1))
    print(' * AUC {:.4f}'.format(AUC))


    if epoch >= 0:
        global_step = epoch * epoch_len + epoch_len - 1
        writer.add_scalar('val_loss', losses.avg, global_step)
        writer.add_scalar('val_acc', accMeter.avg, global_step)
        writer.add_scalar('val_prec', Prec, global_step)
        writer.add_scalar('val_recall', Recall, global_step)
        writer.add_scalar('val_F1', F1, global_step)
        writer.add_scalar('val_AUC', AUC, global_step)

        print('Saving output:')
        np.save(os.path.join(args.output_path, 'out{:0>3}.npy'.format(epoch)), out_array)

    return accMeter.avg,Prec,Recall,F1,AUC


def adjust_learning_rate(optimizer, epoch, file_path):
    f = open(file_path)
    lines = f.readlines()
    lines = [i.strip() for i in lines]
    lines = [i for i in lines if i]
    f.close()

    tmp_lr = 0.00001
    for line in lines:
        t, l = line.split()
        t = int(t)
        l = float(l)
        tmp_lr = l

        if epoch < t:
            lr = l
            break
    else:
        lr = tmp_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    global_step = epoch * epoch_len
    print('epoch' + str(epoch) + ' learning rate: ' + str(lr) + '\n')
    writer.add_scalar('lr', lr, global_step)

def accuracy_old(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy(output, target, threshold):
    logits = torch.sigmoid(output) #torch.float32
    logits = logits.view(logits.size(0))
    logits_n = logits.detach().cpu().numpy() #np.float32
    logits_n = (logits_n >= threshold).astype(np.int64) #np.int64
    target_n = target.detach().cpu().numpy() #np.int64
    correct_num = np.sum((logits_n == target_n).astype(np.int64)) #np.int64
    acc = correct_num/output.size(0) #np.float64
    acc = float(acc) #float
    return acc

def precision_recall_F1(output, target, threshold):
    '''
    :param output:shape (N,), no sigmoid, numpy array, float32
    :param target: shape(N,), numpy array int64
    :param threshold: python number
    :return: precision recall F1  all float python number
    '''
    logits_n = special.expit(output)
    logits_n = (logits_n >= threshold).astype(np.int64)  # np.int64
    target_n = target  # np.int64

    TP = np.sum(target_n.astype(np.bool) & logits_n.astype(np.bool))

    precision = float(TP/np.sum(logits_n))
    recall = float(TP/np.sum(target_n))
    F1 = 2*precision*recall / (precision + recall)

    return precision, recall, F1

def AUROC(output, target):
    logits_n = special.expit(output)
    target_n = target # np.int64
    auroc = roc_auc_score(target_n, logits_n)
    return auroc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, epoch):
    if is_best:
        filepath = os.path.join(args.checkpoint_path, 'model{:0>3}best.pth'.format(epoch))
        torch.save(state, filepath)
    elif epoch % args.save_freq == 0:
        filepath = os.path.join(args.checkpoint_path, 'model{:0>3}.pth'.format(epoch))
        torch.save(state, filepath)

if __name__ == '__main__':
    main()