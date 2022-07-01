from __future__ import print_function
import os
import time
import numpy as np
import warnings
import argparse

import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F

from dataloader import load_data
from utils import *
from taskcv_loader import CVDataLoader
from basenet import *
from torchvision import transforms, datasets

warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--num_classes', type=int, default=12, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--seed', type=int, default=3289761323, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=4, metavar='K',
                    help='how many steps to repeat the generator update')
parser.add_argument('--num-layer', type=int, default=2, metavar='K',
                    help='how many layers for classifier')
parser.add_argument('--num_gpu', type=int, default=1, metavar='B',
                    help='board dir')
parser.add_argument('--save', type=str, default='save/mcd', metavar='B',
                    help='board dir')
parser.add_argument('--train_path', type=str, default='/data1/TL/data//visda2017/clf/train', metavar='B',
                    help='directory of source datasets')
parser.add_argument('--val_path', type=str, default='/data1/TL/data/visda2017/clf/validation', metavar='B',
                    help='directory of target datasets')
parser.add_argument('--load_weight', type=str, default='101', metavar='B',
                    help='which resnet 18,50,101,152,200')
parser.add_argument('--resnet', type=str, default='101', metavar='B',
                    help='which resnet 18,50,101,152,200')

args = parser.parse_args()
args.cuda = True
train_path = args.train_path
val_path = args.val_path
num_k = args.num_k
num_layer = args.num_layer
batch_size = args.batch_size
save_path = args.save + '_' + str(args.num_k)

# %% load_data part
data_transforms = {
    train_path: transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    val_path: transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
dsets = {x: datasets.ImageFolder(os.path.join(x), data_transforms[x]) for x in [
    train_path, val_path]}
dset_sizes = {x: len(dsets[x]) for x in [train_path, val_path]}
dset_classes = dsets[train_path].classes
dset_classes = [i for i in range(args.num_classes)]
classes_acc = {}
for i in dset_classes:
    classes_acc[i] = []
    classes_acc[i].append(0)
    classes_acc[i].append(0)
print('classes' + str(dset_classes))
print('lr', args.lr)
use_gpu = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

source_loader, target_loader, val_loader = load_data(train_path=args.train_path, val_path=args.val_path, batch_size=batch_size)

# %% load model part
option = 'resnet' + args.resnet
if args.resnet == '101':
    source_G = ResBottle(option)
    source_F1 = ResClassifier(num_classes=args.num_classes,
                            num_layer=num_layer, num_unit=source_G.output_num(), middle=1000)
    source_F2 = ResClassifier(num_classes=args.num_classes,
                            num_layer=num_layer, num_unit=source_G.output_num(), middle=1000)
else:
    source_G = ResBottle(option)
    source_F1 = ResClassifier_office(num_classes=args.num_classes,
                            num_layer=num_layer, num_unit=source_G.output_num(), middle=1000)
    source_F2 = ResClassifier_office(num_classes=args.num_classes,
                            num_layer=num_layer, num_unit=source_G.output_num(), middle=1000)
if args.resnet == '101':
    G = ResBottle(option)
    F1 = ResClassifier(num_classes=args.num_classes, 
                    num_layer=num_layer, num_unit=G.output_num(), middle=1000)
    F2 = ResClassifier(num_classes=args.num_classes,
                    num_layer=num_layer, num_unit=G.output_num(), middle=1000)
else:
    G = ResBottle(option)
    F1 = ResClassifier_office(num_classes=args.num_classes, 
                    num_layer=num_layer, num_unit=G.output_num(), middle=1000)
    F2 = ResClassifier_office(num_classes=args.num_classes,
                    num_layer=num_layer, num_unit=G.output_num(), middle=1000)
lr = args.lr 

if args.load_weight != '':
    print("load weight from : ", args.load_weight)
    model = torch.load(args.load_weight)
    G.load_state_dict(model['net_G'])
    F1.load_state_dict(model['net_F1'])
    F2.load_state_dict(model['net_F2'])
    source_G.load_state_dict(model['net_G'])
    source_F1.load_state_dict(model['net_F1'])
    source_F2.load_state_dict(model['net_F2'])

if args.optimizer == 'momentum':
    optimizer_g = optim.SGD(list(G.parameters()),
                            lr=args.lr,  momentum=0.9, weight_decay=0.0005)
    optimizer_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=0.9, lr=args.lr * 10 / 3,
                            weight_decay=0.0005)

if args.cuda:
    devices = [i for i in range(args.num_gpu)]
    G = torch.nn.DataParallel(G.cuda(), device_ids=devices)
    F1 = torch.nn.DataParallel(F1.cuda(), device_ids=devices)
    F2 = torch.nn.DataParallel(F2.cuda(), device_ids=devices)
    source_G = torch.nn.DataParallel(source_G.cuda(), device_ids=devices)
    source_F1 = torch.nn.DataParallel(source_F1.cuda(), device_ids=devices)
    source_F2 = torch.nn.DataParallel(source_F2.cuda(), device_ids=devices)
    
for k, v in source_G.named_parameters():
    v.requires_grad = False
for k, v in source_F1.named_parameters():
    v.requires_grad = False
for k, v in source_F2.named_parameters():
    v.requires_grad = False

def get_distance_T(output1, output2):
    dist = []
    for k in range(output1.size(0)):
        output_t1, output_t2 = output1[k].unsqueeze(0), output2[k].unsqueeze(0)
        dist.append(cdd(output_t1, output_t2))
    return torch.tensor(dist)  

def train(num_epoch):
    criterion = soft_criterion
    best_acc = 0
    gamma = 0.05
    eta = 0.0025
    beta = 0.1
    source_G.eval()
    source_F1.eval()
    source_F2.eval()
    temp_acc = test(0)
    for ep in range(num_epoch):
        since = time.time()
        for batch_idx, (images, _, _) in enumerate(target_loader):
            G.train()
            F1.train()
            F2.train()
            if batch_idx * batch_size > 30000:
                break

            if args.resnet == '50':
                adjust_learning_rate(optimizer_f, ep, batch_idx, 44, 0.01)
            elif args.resnet == '101':
                adjust_learning_rate(optimizer_f, ep, batch_idx, 30000 // batch_size, 0.001)

            if args.cuda:
                images = images.cuda()

            """source domain discriminative"""
            with torch.no_grad():
                src_feature = source_G(images)
                src_output1, src_output2 = source_F1(src_feature), source_F2(src_feature)
                src_log_output1, src_log_output2 = F.softmax(
                    src_output1, dim=1), F.softmax(src_output2, dim=1)

            # Step A train all networks to minimize loss on source
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            
            tgt_feature = G(images)
            tgt_output1, tgt_output2 = F1(tgt_feature), F2(tgt_feature)
            tgt_log_output1, tgt_log_output2 = F.softmax(
                tgt_output1, dim=1), F.softmax(tgt_output2, dim=1)
            entropy_loss = - \
                torch.mean(torch.log(torch.mean(tgt_log_output1, 0) + 1e-6))
            entropy_loss -= torch.mean(
                torch.log(torch.mean(tgt_log_output2, 0) + 1e-6))
            loss1 = criterion(src_log_output1, tgt_log_output1)
            loss2 = criterion(src_log_output2, tgt_log_output2)
            stepA_loss = beta * (loss1 + loss2) + gamma * entropy_loss
            stepA_loss.backward()
            optimizer_g.step()
            optimizer_f.step()

            # # # Step B train all networks to minimize loss on source
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            tgt_feature = G(images)
            tgt_output1, tgt_output2 = F1(tgt_feature), F2(tgt_feature)
            tgt_log_output1, tgt_log_output2 = F.softmax(
                tgt_output1, dim=1), F.softmax(tgt_output2, dim=1)
            entropy_loss = - \
                torch.mean(torch.log(torch.mean(tgt_log_output1, 0) + 1e-6))
            entropy_loss -= torch.mean(
                torch.log(torch.mean(tgt_log_output2, 0) + 1e-6))

            loss1 = criterion(src_log_output1, tgt_log_output1)
            loss2 = criterion(src_log_output2, tgt_log_output2)

            cdd_dist = cdd(tgt_log_output1, tgt_log_output2)

            stepB_loss = beta * (loss1 + loss2) - eta * cdd_dist + gamma * entropy_loss
            stepB_loss.backward()
            optimizer_f.step()

            # Step C train all networks to minimize loss on source
            for i in range(num_k):
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()
                tgt_feature = G(images)
                tgt_output1, tgt_output2 = F1(tgt_feature), F2(tgt_feature)
                tgt_log_output1, tgt_log_output2 = F.softmax(
                    tgt_output1, dim=1), F.softmax(tgt_output2, dim=1)
                entropy_loss = -torch.mean(torch.log(torch.mean(
                        tgt_log_output1, 0) + 1e-6))
                entropy_loss -= torch.mean(
                    torch.log(torch.mean(tgt_log_output2, 0) + 1e-6))
                cdd_dist = cdd(tgt_log_output1, tgt_log_output2)

                stepC_loss = eta * cdd_dist + gamma * entropy_loss

                stepC_loss.backward()
                optimizer_g.step()

            if batch_idx % args.log_interval == 0:
                print(
                    'Train Ep: {} [{}/{} ({:.6f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} Entropy: {:.6f} '.format(
                        ep, batch_idx, 30000 // batch_size, 100. * batch_idx / (30000 // batch_size),
                        loss1.item(), loss2.item(), cdd_dist.item(), entropy_loss.item()))

        # test
        temp_acc = test(ep + 1)
        # if temp_acc > best_acc:
        best_acc = temp_acc
        best_dict = {
            'net_G': G.module.state_dict(),
            'net_F1': F1.module.state_dict(),
            'net_F2': F2.module.state_dict()
        }
        print('save model to ', save_path + 'pth')
        torch.save(best_dict, save_path + '.pth')
        print('\tbest:', best_acc)
        print('time:', time.time() - since)
        print('-' * 100)
    
def test(epoch):
    G.eval()
    F1.eval()
    F2.eval()
    test_loss = 0
    correct_add = 0
    size = 0
    start_test = 0
    print('-' * 100, '\nTesting')
    with torch.no_grad():
        for batch_idx, (img, label, _ ) in enumerate(val_loader):
            if batch_idx * batch_size > 5000:
                break
            if args.cuda:
                img, label = img.cuda(), label.cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            output = G(img)
            output1 = F1(output)
            output2 = F2(output)
            dist = get_distance_T(F.softmax(output1, dim=1), F.softmax(output2, dim=1))
            test_loss += F.nll_loss(output1, label).item()
            output_add = output1 + output2
            pred = output_add.data.max(1)[1]
            correct_add += pred.eq(label.data).cpu().sum()
            size += label.data.size()[0]
            for i in range(len(label)):
                key_label = dset_classes[label.long()[i].item()]
                key_pred = dset_classes[pred.long()[i].item()]
                classes_acc[key_label][1] += 1
                if key_pred == key_label:
                    classes_acc[key_pred][0] += 1

    # loss function already averages over batch size
    print('Epoch: {:d} Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)'.format(
        epoch, test_loss, correct_add, size, 100. * float(correct_add) / size))
    avg = []
    for i in dset_classes:
        print('\t{}: [{}/{}] ({:.6f}%)'.format(i, classes_acc[i][0], classes_acc[i][1],
                                               100. * classes_acc[i][0] / classes_acc[i][1]))
        avg.append(100. * float(classes_acc[i][0]) / classes_acc[i][1])
    temp_acc = np.average(avg)
    print('\taverage:', temp_acc)
    for i in dset_classes:
        classes_acc[i][0] = 0
        classes_acc[i][1] = 0
    return temp_acc


train(args.epochs + 1)
