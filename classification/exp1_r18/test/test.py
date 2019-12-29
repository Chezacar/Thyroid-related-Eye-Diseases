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

parser = argparse.ArgumentParser(description='TAO testing')
parser.add_argument('--workers',default=6,type=int, metavar='N')
parser.add_argument('-b', '--batch_size',default=64, type=int,metavar='N')
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-op', '--output_path', default='', type=str, metavar='PATH', help='path to store test output')
parser.add_argument('--suffix', default ='', type = str, help = 'suffix of summmary and checkpoint dir')
parser.add_argument('--threshold',default=0.5,type = float, metavar='THRESHOLD')
parser.add_argument('--test_file', default='/DB/rhome/zdcheng/workspace/hyperthyreosis_eye/classification/all_test.json',
                    type=str, metavar='PATH', help='path to val json file')
parser.add_argument('--data_root',default='/DATA5_DB8/data/zdcheng/hyperthyreosis_eye/class_arraydata', type = str,
                    metavar='PATH',help='path to image data root')

def main():
    global args, writer
    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.suffix)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    model = resnet.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 1)
    model = torch.nn.DataParallel(model).cuda()

    criterion = nn.BCEWithLogitsLoss()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise RuntimeError()
    test_dataset = TAO_loader.TAO(args.test_file, args.data_root, mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.workers, pin_memory=True)
    test(test_loader, model, criterion)
def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accMeter = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
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
                i, len(test_loader), batch_time=batch_time, loss=losses,
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

    print('Saving output:')
    np.save(os.path.join(args.output_path, 'out.npy'), out_array)
    np.save(os.path.join(args.output_path, 'target.npy'), target_array)


    return accMeter.avg,Prec,Recall,F1,AUC

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


if __name__ == '__main__':
    main()