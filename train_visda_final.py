from __future__ import print_function
import os
import time
import numpy as np
import warnings
import argparse
import shutil

import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import data_transformes
import random 
# from data_pipline import get_data_loader
from utils import *
from taskcv_loader import CVDataLoader
from basenet import *
from torchvision import transforms, datasets
from dataloader import load_data

warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--num_classes', type=int, default=12, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--seed', type=int, default=19990817, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=4, metavar='K',
                    help='how many steps to repeat the generator update')
parser.add_argument('--num-layer', type=int, default=2, metavar='K',
                    help='how many layers for classifier')
parser.add_argument('--num_gpu', type=int, default=4, metavar='B',
                    help='board dir')
parser.add_argument('--save', type=str, default='save/mcd', metavar='B',
                    help='board dir')
parser.add_argument('--train_path', type=str, default='/data1/TL/data//visda2017/clf/train', metavar='B',
                    help='directory of source datasets')
parser.add_argument('--val_path', type=str, default='/data1/TL/data/visda2017/clf/validation', metavar='B',
                    help='directory of target datasets')
parser.add_argument('--resnet', type=str, default='101', metavar='B',
                    help='which resnet 18,50,101,152,200')
parser.add_argument('--soft', type=int, default=0, metavar='B',
                    help='which resnet 18,50,101,152,200')
parser.add_argument('--load_weight', type=str, default='')
parser.add_argument('--topk', type=int, default=9)
parser.add_argument('--load_weight_source', type=str, default='')
parser.add_argument('--augment', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--flag', type=int, default=0)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

args = parser.parse_args()
args.cuda = True
setup_seed(args.seed)

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
dsets = {x: datasets.ImageFolder(os.path.join(x), data_transforms[x]) for x in [train_path, val_path]}
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
    model = torch.load(args.load_weight)
    source_G.load_state_dict(model['net_G'])
    source_F1.load_state_dict(model['net_F1'])
    source_F2.load_state_dict(model['net_F2'])

if args.load_weight_source != '':
    model = torch.load(args.load_weight_source)
    G.load_state_dict(model['net_G'])
    F1.load_state_dict(model['net_F1'])
    F2.load_state_dict(model['net_F2']) 

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

torch.backends.cudnn.benchmark = True
    
def get_distance_T(output1, output2):
    dist = []
    for k in range(output1.size(0)):
        output_t1, output_t2 = output1[k].unsqueeze(0), output2[k].unsqueeze(0)
        dist.append(cdd(output_t1, output_t2))
    return torch.tensor(dist)        

def split_images_perclass_v3(data_path, ratio=0.5, alpha=0.0):
    _, _, val_loader = load_data(train_path=data_path, val_path=data_path, batch_size=128)
    class_name = os.listdir(data_path)
    class_name.sort()
    start_train = 0
    source_G.eval()
    source_F1.eval()
    source_F2.eval()
    all_path = []
    softlabel_map = {}
    with torch.no_grad():
        for batch_idx, (images, label, path) in enumerate(val_loader):
            for img_path in path:
                all_path.append(img_path)
            images, label = images.cuda(), label.cuda()
            tgt_feature = source_G(images)
            tgt_output1, tgt_output2 = source_F1(tgt_feature), source_F2(tgt_feature)
            tgt_log_output1, tgt_log_output2 = F.softmax(
                tgt_output1, dim=1), F.softmax(tgt_output2, dim=1)
            dist = get_distance_T(tgt_log_output1, tgt_log_output2)
            soft_pred = (tgt_log_output1 + tgt_log_output2) / 2
            
            for idx, img_path in enumerate(path):
                part = img_path.split('/')
                softlabel_map[part[-1]] = soft_pred[idx].detach().cpu().float().unsqueeze(0)

            if start_train == 0:
                start_train = 1
                all_fea = tgt_feature.detach().cpu().float()
                all_prob1 = tgt_log_output1.cpu().float()
                all_prob2 = tgt_log_output2.cpu().float()
                all_label = label.detach().cpu().long()
            else:
                all_fea = torch.cat((all_fea, tgt_feature.detach().cpu().float()), 0)
                all_prob1 = torch.cat((all_prob1, tgt_log_output1.detach().cpu().float()), 0)
                all_prob2 = torch.cat((all_prob2, tgt_log_output2.detach().cpu().float()), 0)
                all_label = torch.cat((all_label, label.detach().cpu().long()), 0)

    ensemble_prob1 = all_prob1
    ensemble_prob2 = all_prob2
    ensemble_pred = (ensemble_prob1 + ensemble_prob2) / 2
    all_pred = ensemble_pred.max(1)[1].detach().cpu().long()
    all_dist = get_distance_T(ensemble_prob1, ensemble_prob2)

    src_data_path = './data/split_source_data/'
    tgt_data_path = './data/split_target_data/'
    
    if os.path.exists(src_data_path):
        shutil.rmtree(src_data_path)
        os.makedirs(src_data_path)

    if os.path.exists(tgt_data_path):
        shutil.rmtree(tgt_data_path)        
        os.makedirs(tgt_data_path)

    ave_acc = 0
    expect_num = all_dist.shape[0] / args.num_classes
    min_num, max_num = int(expect_num * (ratio - 0.4)), int(expect_num * (ratio + 0.4))
    for i in range(args.num_classes):
        now_index = torch.where(all_pred == i)[0] 
        limit = int(len(all_dist[now_index]) * ratio ) 
        limit = max(limit, min_num)
        limit = min(limit, max_num)
        sim_index_i = all_dist[now_index].argsort()[:limit]
        sim_index_i = now_index[sim_index_i]
        if i == 0:
            sim_index = sim_index_i.detach().cpu().long()
        else:
            sim_index = torch.cat((sim_index, sim_index_i.detach().cpu().long()), 0)
        acc =  ( all_pred[sim_index_i].eq(all_label[sim_index_i]).sum().float() / len(sim_index_i)).item()
        print("Accuracy : ",  acc) 
        if len(sim_index_i) != 0 : ave_acc += acc
        print(len(sim_index_i))
    print("Ave Accuracy : ", ave_acc / args.num_classes)    

    for file_name in class_name:
        save_path = src_data_path + file_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
    for file_name in class_name:
        save_path = tgt_data_path + file_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    cnt = 0
    for batch_idx, img_path in enumerate(all_path):
        idx = torch.where(sim_index == batch_idx)[0]
        if idx.numel() != 0: 
            save_idx = sim_index[idx]
            save_path = src_data_path + class_name[all_pred[save_idx]]
        else:
            cnt += 1
            save_idx = int(torch.rand(1) * args.num_classes)
            save_path = tgt_data_path + class_name[all_pred[save_idx]]
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        shutil.copy(img_path, save_path)
    print(sim_index.shape[0])
    print('source images : ', len(all_path) - cnt, 'target images :', cnt)            
    return softlabel_map


def train(num_epoch):
    source_aug = data_transformes.data_transformes()

    criterion = soft_criterion_weight
    best_acc = 0
    beta = 1
    tgt_aug = False
    gamma = 0.01
    eta = 0.0025
    alpha = 0.95
    source_G.eval()
    source_F1.eval()
    source_F2.eval()
    test(0, source_G, source_F1, source_F2)
    
    for ep in range(num_epoch):
        
        since = time.time()
        soft_labels = split_images_perclass_v3(args.val_path, ratio=0.5)
        src_data_path = './data/split_source_data/'
        tgt_data_path = './data/split_target_data/'
        source_loader, target_loader, _ = load_data(train_path=src_data_path, val_path=tgt_data_path, batch_size=batch_size)

        for batch_idx, ( (src_images, src_label , path), (tgt_images, _, _ ) ) in enumerate(zip(source_loader, target_loader)):
            G.train()
            F1.train()
            F2.train()
            
            if args.augment:
                src_images = source_aug.augment(src_images.numpy())
                src_images = torch.autograd.Variable(torch.from_numpy(src_images))
                if tgt_aug:
                    aug_tgt_images = source_aug.augment(tgt_images.numpy())
                    aug_tgt_images = torch.autograd.Variable(torch.from_numpy(aug_tgt_images))


            if batch_idx * batch_size > 30000:
                break

            if args.soft :
                for idx, img_path in enumerate(path):
                    key = img_path.split('/')[-1]
                    try:
                        soft_label = torch.cat((soft_label, soft_labels[key]), 0)
                    except:
                        soft_label = soft_labels[key] 

            adjust_learning_rate(optimizer_f, ep, batch_idx, 30000 // batch_size, 0.001)
            # adjust_learning_rate(optimizer_f, ep, batch_idx, 45, 0.01)

            if args.cuda:
                tgt_images = tgt_images.cuda()
                if args.soft : soft_label = soft_label.cuda()
                src_images, label = src_images.cuda(), src_label.cuda()
            
            if args.soft:
                with torch.no_grad():
                    feature = source_G(src_images)
                    weight = 1 - get_distance_T( F.softmax(source_F1(feature), dim=1) , F.softmax(source_F2(feature), dim=1) )
                    weight = 1.0 * F.sigmoid(weight).unsqueeze(1).cuda()
                    print(weight)

            """source domain discriminative"""
            # Step A train all networks to minimize loss on source
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            src_feature = G(src_images)
            src_output1, src_output2 = F1(src_feature), F2(src_feature)
            src_log_output1, src_log_output2 = F.softmax(
                src_output1, dim=1), F.softmax(src_output2, dim=1)
            tgt_feature = G(tgt_images)
            tgt_output1, tgt_output2 = F1(tgt_feature), F2(tgt_feature)
            tgt_log_output1, tgt_log_output2 = F.softmax(
                tgt_output1, dim=1), F.softmax(tgt_output2, dim=1)
            entropy_loss = - \
                torch.mean(torch.log(torch.mean(tgt_log_output1, 0) + 1e-6))
            entropy_loss -= torch.mean(
                torch.log(torch.mean(tgt_log_output2, 0) + 1e-6))
            if args.soft:
                loss1 = criterion(src_log_output1, soft_label, weight)
                loss2 = criterion(src_log_output2, soft_label, weight)
            else:
                loss1 = beta * torch.nn.CrossEntropyLoss()(src_output1, label)
                loss2 = beta * torch.nn.CrossEntropyLoss()(src_output2, label)
            
            stepA_loss =  (loss1 + loss2) + gamma * entropy_loss
            if tgt_aug:
                aug_tgt_feature = G(aug_tgt_images)
                aug_tgt_output1, aug_tgt_output2 = F1(aug_tgt_feature), F2(aug_tgt_feature)
                aug_tgt_log_output1, aug_tgt_log_output2 = F.softmax(
                    aug_tgt_output1, dim=1), F.softmax(aug_tgt_output2, dim=1)
                aug_loss = torch.mean(torch.abs(aug_tgt_log_output1 - tgt_log_output1))             
                aug_loss += torch.mean(torch.abs(aug_tgt_log_output2 - tgt_log_output2))
                stepA_loss += aug_loss

            stepA_loss.backward()

            optimizer_g.step()
            optimizer_f.step()

            # Step B train all networks to minimize loss on source
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            src_feature = G(src_images)
            src_output1, src_output2 = F1(src_feature), F2(src_feature)
            src_log_output1, src_log_output2 = F.softmax(
                src_output1, dim=1), F.softmax(src_output2, dim=1)

            tgt_feature = G(tgt_images)
            tgt_output1, tgt_output2 = F1(tgt_feature), F2(tgt_feature)
            tgt_log_output1, tgt_log_output2 = F.softmax(
                tgt_output1, dim=1), F.softmax(tgt_output2, dim=1)
            entropy_loss = - \
                torch.mean(torch.log(torch.mean(tgt_log_output1, 0) + 1e-6))
            entropy_loss -= torch.mean(
                torch.log(torch.mean(tgt_log_output2, 0) + 1e-6))

            if args.soft:
                loss1 = criterion(src_log_output1, soft_label, weight)
                loss2 = criterion(src_log_output2, soft_label, weight)
            else:
                loss1 = beta * torch.nn.CrossEntropyLoss()(src_output1, label)
                loss2 = beta * torch.nn.CrossEntropyLoss()(src_output2, label)

            cdd_dist = cdd(tgt_log_output1, tgt_log_output2)
            stepB_loss = (loss1 + loss2) - eta * cdd_dist + gamma * entropy_loss
            stepB_loss.backward()
            optimizer_f.step()

            # Step C train all networks to minimize loss on source
            for i in range(num_k):
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()
                tgt_feature = G(tgt_images)
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

            # scheduler_f.step()

            if batch_idx % args.log_interval == 0:
                print(
                    'Train Ep: {} [{}/{} ({:.6f}%)] Loss1: {:.6f} Loss2: {:.6f}  Dis: {:.6f} Entropy: {:.6f} '.format(
                        ep, batch_idx, 30000 // batch_size , 100. * batch_idx /  (30000 // batch_size),
                        loss1.item(), loss2.item(), cdd_dist.item(), entropy_loss.item()))
                # test
        if loss1 > 0.5 and loss2 > 0.5:
            for mean_param, param in zip(source_G.module.parameters(), G.module.parameters()):
                mean_param.data = mean_param.data * alpha + (1 - alpha) * param.data 
            for mean_param, param in zip(source_F1.parameters(), F1.parameters()):
                mean_param.data = mean_param.data * alpha + (1 - alpha) * param.data 
            for mean_param, param in zip(source_F2.parameters(), F2.parameters()):
                mean_param.data = mean_param.data * alpha + (1 - alpha) * param.data 
        else:
            source_G.module.load_state_dict(G.module.state_dict())
            source_F2.module.load_state_dict(F2.module.state_dict())
            source_F1.module.load_state_dict(F1.module.state_dict())
        
        temp_acc = test(ep + 1, G, F1, F2)
        if temp_acc > best_acc :
            best_acc = temp_acc
            best_dict = {
                'net_G': G.module.state_dict(),
                'net_F1': F1.module.state_dict(),
                'net_F2': F2.module.state_dict()
            }
            torch.save(best_dict, save_path + '.pth')
        print('\tbest:', best_acc)
        print('time:', time.time() - since)
        print('-' * 100)

def test(epoch, G, F1, F2):
    G.eval()
    F1.eval()
    F2.eval()
    test_loss = 0
    correct_add = 0
    size = 0
    start_test = 0
    print('-' * 100, '\nTesting')
    _, target_loader, val_loader = load_data(train_path=args.val_path, val_path=args.val_path, batch_size=batch_size)
    with torch.no_grad():
        for batch_idx, (img, label, _) in enumerate(val_loader):
            if batch_idx * batch_size > 5000:
                break
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
            if start_test == 0:
                start_test = 1
                all_fea = output_add.cpu().float()
                all_label = label.cpu().long()
            else:
                all_fea = torch.cat((all_fea, output_add.cpu().float()), 0)
                all_label = torch.cat((all_label, label.cpu().long()), 0)

    draw_data(all_fea, all_label, name='final')

    # loss function already averages over batch size
    print('Epoch: {:d} Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)'.format(
        epoch, test_loss, correct_add, size, 100. * float(correct_add) / size))
    avg = []
    for i in dset_classes:
        print('\t{}: [{}/{}] ({:.6f}%)'.format(i, classes_acc[i][0], classes_acc[i][1],
                                               100. * classes_acc[i][0] / classes_acc[i][1]))
        if classes_acc[i][1] != 0:
            avg.append(100. * float(classes_acc[i][0]) / classes_acc[i][1])
        else:
            avg.append(0)
    temp_acc = np.average(avg)
    print('\taverage:', temp_acc)
    for i in dset_classes:
        classes_acc[i][0] = 0
        classes_acc[i][1] = 0
    return temp_acc

train(args.epochs + 1)

# %%
