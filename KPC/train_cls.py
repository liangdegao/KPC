from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from data import ModelNet40
from models.KPC_cls import get_model_cls
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from tqdm import tqdm
from tensorboardX import SummaryWriter
from transform import Compose,RandomRotate,RandomScale,RandomJitter

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    io.cprint("Let's use" + str(torch.cuda.device_count()) + "GPUs!")

    writer = SummaryWriter('./check/log')

    inchannel = [64, 128, 256, 512]

    train_transform = Compose([
        RandomRotate(),
        RandomScale(),
        RandomJitter()
    ])

    transform = train_transform

    model = get_model_cls(40, in_channel=inchannel).to(device)  ####
    model = nn.DataParallel(model)
    print('Use pointnet2_cls_test')

    if args.use_sgd:
        io.cprint("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        io.cprint("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-6)
    elif args.scheduler == 'step':
        scheduler = MultiStepLR(opt, [120, 160], gamma=0.1)

    criterion = cal_loss

    best_train_acc = 0
    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []

        t = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch')

        for data, label,group in t:
            data1 = data.clone()
            data1 = np.array(data1)
            if transform:
                data1, data1 = transform(data1, data1)
            data = torch.tensor(data1)
            group = group.view(-1, 16, 40, 4)
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data,group)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            t.set_postfix_str(f"loss: {loss:.8f}")
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        writer.add_scalar('loss', train_loss * 1.0 / count, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, best train acc: %.6f' % (epoch, train_loss * 1.0 / count,
                                                            metrics.accuracy_score(
                                                                train_true, train_pred), best_train_acc)
        io.cprint(outstr)

        if np.mean(train_acc) > best_train_acc:
            best_train_acc = np.mean(train_acc)
            torch.save(model.state_dict(), f"./check/r_pointnet_cls/best_model_cls.pth")
            io.cprint("Model saved!")

        if epoch % 5 == 0:
            ####################
            # Test
            ####################
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []
            for data, label in tqdm(test_loader, desc="testing"):
                # group = group.view(-1, 16, 6, 3)
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits = model(data)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            writer.add_scalar('test_acc', test_acc, epoch)
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, best test acc: %.6f' % (epoch, test_loss * 1.0 / count, test_acc, best_test_acc)
            io.cprint(outstr)
            if np.mean(test_acc) > best_test_acc:
                best_test_acc = np.mean(test_acc)
                torch.save(model.state_dict(), f"./check/r_pointnet_cls/best_model_test_cls.pth")
                io.cprint("ModelTest saved!")

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=128, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    seed = np.random.randint(1, 10000)

    if args.eval:
        io = IOStream('../check/' + args.exp_name + '/eval.log')
    else:
        io = IOStream('../check/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    io.cprint('random seed is: ' + str(seed))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    else:
        io.cprint('Using CPU')

    train(args, io)

