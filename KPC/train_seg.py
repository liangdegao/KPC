from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR
from models.KPC_seg import get_model
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream, focal_loss
import sklearn.metrics as metrics
from data import ShapeNetPart
from tqdm import tqdm
from tensorboardX import SummaryWriter
from transform import Compose,RandomRotate,RandomScale,RandomJitter
from einops import rearrange

seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

def calculate_shape_IoU(pred_np, seg_np, label, class_choice, eva=False):
    label = label.squeeze()
    shape_ious = []
    category = {}
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
        if label[shape_idx] not in category:
            category[label[shape_idx]] = [shape_ious[-1]]
        else:
            category[label[shape_idx]].append(shape_ious[-1])

    if eva:
        return shape_ious, category
    else:
        return shape_ious

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    device = output.device
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    target = target.to(device)
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def train(args, io):
    train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_points, class_choice=args.class_choice)
    if (len(train_dataset) < 100):
        drop_last = False
    else:
        drop_last = True
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True,
                              drop_last=drop_last)
    test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice),
                             num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=drop_last)

    writer = SummaryWriter('./check/log')

    device = torch.device("cuda" if args.cuda else "cpu")
    io.cprint("Let's use" + str(torch.cuda.device_count()) + "GPUs!")

    seg_num_all = train_loader.dataset.seg_num_all
    seg_start_index = train_loader.dataset.seg_start_index
    seg_start_index_1 = test_loader.dataset.seg_start_index

    inchannel = [64,128,256,512]
    print("use transform")
    train_transform = Compose([
        RandomRotate(),
        RandomScale(),
        RandomJitter()
    ])

    transform = train_transform

    # create model
    model = get_model(50,in_channel = inchannel).to(device)
    model = nn.DataParallel(model)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-7)
    elif args.scheduler == 'step':
        scheduler = MultiStepLR(opt, [140, 180], gamma=0.1)
    criterion = cal_loss

    best_train_iou = 0
    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []

        total_intersection = np.zeros(50)  # 总交集
        total_union = np.zeros(50)  # 总并集
        total_target = np.zeros(50)

        t = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch')

        for data, label, seg, group, group_1 in t:
            group = rearrange(group, 'B b C CN CP -> (B b) C CN CP').contiguous()
            data1 = data.clone()
            data1 = np.array(data1)
            if transform:
                data1, data1 = transform(data1, data1)
            data = torch.tensor(data1)
            seg = seg - seg_start_index
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            x = model(data, group)
            x = x.permute(0, 1, 2).contiguous()
            loss_1 = criterion(x.view(-1, seg_num_all), seg.view(-1, 1).squeeze())
            focal_1 = focal_loss(x, seg)
            loss = loss_1 + 0.2 * focal_1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            seg_pred = x.clone()
            pred = seg_pred.max(dim=2)[1]  # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            intersection, union, target_area = intersectionAndUnionGPU(pred, seg, 50)
            total_intersection += intersection.cpu().numpy()
            total_union += union.cpu().numpy()
            total_target += target_area.cpu().numpy()
            seg_np = seg.cpu().numpy()  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()  # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))  # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))  # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_label_seg.append(label.reshape(-1))
            writer.add_scalar('Focal_Loss', focal_1)
            writer.add_scalar('loss1', loss_1)
            t.set_postfix_str(f"loss: {loss:.8f}")

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label_seg = np.concatenate(train_label_seg)
        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, args.class_choice)
        train_ious_t = total_intersection / (total_union + 1e-10)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f,train iou t: %.6f' % (epoch,
                                                                                                  train_loss * 1.0 / count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious),
                                                                                                  np.mean(train_ious_t))

        # Log metrics to TensorBoard
        # writer.add_scalar('Focal_Loss', focal_1, epoch)
        writer.add_scalar('Train/Loss', train_loss * 1.0 / count, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)
        writer.add_scalar('Train/AvgClassAccuracy', avg_per_class_acc, epoch)
        writer.add_scalar('Train/mIoU', np.mean(train_ious), epoch)

        io.cprint(outstr)

        if np.mean(train_ious) > best_train_iou:
            best_train_iou = np.mean(train_ious)
            torch.save(model.state_dict(), f"./check/r_pointnet_1/best_model.pth")
            io.cprint("Model saved!")

        if epoch % 5 == 0:
            ####################
            # Test
            ####################
            print("start test")
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            test_label_seg = []
            total_intersection_test = np.zeros(50)  # 总交集
            total_union_test = np.zeros(50)  # 总并集
            total_target_test = np.zeros(50)
            with torch.no_grad():
                for data, label, seg, group, group_1 in tqdm(test_loader, desc="testing"):
                    group = rearrange(group, 'B b C CN CP -> (B b) C CN CP').contiguous()
                    seg = seg - seg_start_index_1
                    data = data.permute(0, 2, 1)
                    batch_size = data.size()[0]
                    seg_pred = model(data, group)
                    seg_pred = seg_pred.contiguous()
                    loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1, 1).squeeze())
                    pred = seg_pred.max(dim=2)[1]
                    intersection, union, target_area = intersectionAndUnionGPU(pred, seg, 50)
                    total_intersection_test += intersection.cpu().numpy()
                    total_union_test += union.cpu().numpy()
                    total_target_test += target_area.cpu().numpy()
                    count += batch_size
                    test_loss += loss.item() * batch_size
                    seg_np = seg.cpu().numpy()
                    pred_np = pred.detach().cpu().numpy()
                    test_true_cls.append(seg_np.reshape(-1))
                    test_pred_cls.append(pred_np.reshape(-1))
                    test_true_seg.append(seg_np)
                    test_pred_seg.append(pred_np)
                    test_label_seg.append(label.reshape(-1))
                test_true_cls = np.concatenate(test_true_cls)
                test_pred_cls = np.concatenate(test_pred_cls)
                test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
                avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
                test_true_seg = np.concatenate(test_true_seg, axis=0)
                test_pred_seg = np.concatenate(test_pred_seg, axis=0)
                test_label_seg = np.concatenate(test_label_seg)
                test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
                test_ious_t = total_intersection_test / (total_union_test + 1e-10)
                outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f, best iou %.6f, best iou t: %.6f' % (
                    epoch,
                    test_loss * 1.0 / count,
                    test_acc,
                    avg_per_class_acc,
                    np.mean(
                        test_ious),
                    best_test_iou,
                    np.mean(test_ious_t))
                writer.add_scalar('Test/mIoU', np.mean(test_ious), epoch)
                io.cprint(outstr)
                if np.mean(test_ious) > best_test_iou:
                    best_test_iou = np.mean(test_ious)
                    torch.save(model.state_dict(), f"./check/r_pointnet_1/best_model_test.pth")
                    io.cprint("ModelTest saved!")

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N',
                        choices=['shapenetpart'])
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--batch_size', type=int, default=128, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
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
    parser.add_argument('--num_points', type=int, default=2048,
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

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    else:
        io.cprint('Using CPU')

    train(args, io)
