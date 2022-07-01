from __future__ import print_function
import argparse
import torch.optim as optim
from utils import *
from taskcv_loader import CVDataLoader
from basenet import *
from torchvision import transforms, datasets
from dataloader import load_data

import torch.nn.functional as F
import os
import time
import numpy as np
import warnings

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
parser.add_argument('--seed', type=int, default=3216321, metavar='S',
                    help='random seed (default: 3216321)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
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
parser.add_argument('--resnet', type=str, default='50', metavar='B',
                    help='which resnet 18,50,101,152,200')
parser.add_argument('--load_weight', type=str, default='')

args = parser.parse_args()
torch.manual_seed(1)
torch.cuda.manual_seed_all(2)
# np.random.seed(3)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
args.cuda = True
train_path = args.train_path
val_path = args.val_path
num_k = args.num_k
num_layer = args.num_layer
batch_size = args.batch_size
save_path = args.save + '_' + str(args.num_k)

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
print(train_path)
dsets = {x: datasets.ImageFolder(os.path.join(x), data_transforms[x]) for x in [train_path, val_path]}
dset_sizes = {x: len(dsets[x]) for x in [train_path, val_path]}
dset_classes = dsets[train_path].classes
dset_classes= [i for i in range(args.num_classes)]
classes_acc = {}
for i in dset_classes:
    classes_acc[i] = []
    classes_acc[i].append(0)
    classes_acc[i].append(0)
print('classes' + str(dset_classes))
print('lr', args.lr)
use_gpu = torch.cuda.is_available()

# train_loader = CVDataLoader()
# train_loader.initialize(dsets[train_path], dsets[val_path], batch_size, shuffle=True, drop_last=True)
# dataset = train_loader.load_data()
# test_loader = CVDataLoader()
# opt = args
# test_loader.initialize(dsets[train_path], dsets[val_path], batch_size, shuffle=True, drop_last=False)
# dataset_test = test_loader.load_data()

train_loader, test_loader, val_loader = load_data(train_path, val_path, batch_size, drop_last=True)

option = 'resnet' + args.resnet
if args.resnet == '101':
    G = ResBottle(option)
    F1 = ResClassifier(num_classes=args.num_classes, 
                    num_layer=num_layer, num_unit=G.output_num(), middle=1000)
    F2 = ResClassifier(num_classes=args.num_classes,
                    num_layer=num_layer, num_unit=G.output_num(), middle=1000)
else :
    G = ResBottle(option)
    F1 = ResClassifier_office(num_classes=args.num_classes, 
                    num_layer=num_layer, num_unit=G.output_num(), middle=1000)
    F2 = ResClassifier_office(num_classes=args.num_classes,
                    num_layer=num_layer, num_unit=G.output_num(), middle=1000)

F1.apply(weights_init)
F2.apply(weights_init)
lr = args.lr

if args.load_weight != '':
    model = torch.load(args.load_weight)
    G.load_state_dict(model['net_G'])
    F1.load_state_dict(model['net_F1'])
    F2.load_state_dict(model['net_F2'])

if args.optimizer == 'momentum':
    optimizer_g = optim.SGD(list(G.parameters()), lr=args.lr,  momentum=0.9, weight_decay=0.0005)
    optimizer_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=0.9, lr=args.lr,
                           weight_decay=0.0005)
elif args.optimizer == 'adam':
    optimizer_g = optim.Adam(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_f = optim.Adam(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)

if args.cuda:
    G = torch.nn.DataParallel(G.cuda())
    F1 = torch.nn.DataParallel(F1.cuda())
    F2 = torch.nn.DataParallel(F2.cuda())


def train(num_epoch):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    best_acc=0
    for ep in range(num_epoch):
        since = time.time()
        

        for batch_idx, (images, label, _) in enumerate(train_loader):
            G.train() 
            F1.train()
            F2.train()
            if args.cuda:
                images, label = images.cuda(), label.cuda()

            """source domain discriminative"""
            # Step A train all networks to minimize loss on source
            adjust_learning_rate(optimizer_f, ep, batch_idx, 45, 0.01)

            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output = G(images)
            output1 = F1(output)
            output2 = F2(output)

            loss1 = criterion(output1, label)
            loss2 = criterion(output2, label)
            all_loss = loss1 + loss2 
            all_loss.backward()
            optimizer_g.step()
            optimizer_f.step()
            
            if batch_idx % args.log_interval == 0:
                print(
                    'Train Ep: {} [{}/{} ({:.6f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t '.format(
                        ep, batch_idx, 45, 100. * batch_idx / 243,
                        loss1.item(), loss2.item()))

        # test
        test(ep)
        # best_acc = temp_acc
        best_dict = {
            'net_G': G.module.state_dict(),
            'net_F1': F1.module.state_dict(),
            'net_F2': F2.module.state_dict()
        }
        print('saved!')
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
        for batch_idx, (img, label, _) in enumerate(val_loader):
            if args.cuda:
                img, label = img.cuda(), label.cuda()

            img, label = Variable(img, volatile=True), Variable(label)
            output = G(img)
            output1 = F1(output)
            output2 = F2(output)
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
    #         if start_test == 0:
    #             start_test = 1
    #             all_fea = output_add.cpu().float()
    #             all_label = label.cpu().long()
    #         else:
    #             all_fea = torch.cat((all_fea, output_add.cpu().float()), 0)
    #             all_label = torch.cat((all_label, label.cpu().long()), 0)

    # draw_data(all_fea, all_label, name='target')

    # loss function already averages over batch size
    test_loss /= len(test_loader)
    print('Epoch: {:d} Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)'.format(
        epoch, test_loss, correct_add, size, 100. * float(correct_add) / size))
    avg = []
    for i in dset_classes:
        # print('\t{}: [{}/{}] ({:.6f}%)'.format(i, classes_acc[i][0], classes_acc[i][1],
        #                                        100. * classes_acc[i][0] / classes_acc[i][1]))
        if classes_acc[i][1] == 0: continue
        avg.append(100. * float(classes_acc[i][0]) / classes_acc[i][1])
    temp_acc = np.average(avg)
    print('\taverage:', temp_acc)
    for i in dset_classes:
        classes_acc[i][0] = 0
        classes_acc[i][1] = 0
    return temp_acc

train(args.epochs + 1)
