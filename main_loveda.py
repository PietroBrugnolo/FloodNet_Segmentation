# -------------------------------------------------------------------------
# Author:   Pietro Brugnolo
# Date:     20/11/2023
# Version:  1
# -------------------------------------------------------------------------

import torch
import os
import sys
import gc
import argparse
import metrics
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image 
from torchvision.models.segmentation import deeplabv3_resnet50
import segmentation_models_pytorch as smp 
from torch.utils.data import DataLoader
from segmentation_dataset import SegmentationDataset
import unets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device, "\n")
# device = torch.device('cpu')

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=40,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of training epochs')
    parser.add_argument('--project_folder', type=str, default=None,
                        help='project folder')

    # optimization
    parser.add_argument('--learning_rate', type=float, default= 0.001,
                        help='learning rate')
    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     help='momentum')
    # model
    parser.add_argument('--name_net', type=str, default='unet', help='name of the network')

    # dataset
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--num_classes', type=int, default=8, help='path to image dataset')

    # # temperature
    # parser.add_argument('--temp', type=float, default=0.07,
    #                     help='temperature for loss function')
  
    # parser.add_argument('--dataset_dim', type=str, default=20, help='path to image dataset')
    parser.add_argument('--train_data_folder', type=str, default=None, help='path to train dataset')
    parser.add_argument('--val_data_folder', type=str, default=None, help='path to val dataset')

    opt = parser.parse_args()

    #----------------------------------------------------------------------------------------
    # # Dataset
    # opt.mean = (0.5139, 0.4424, 0.3937)
    # opt.std = (0.2748, 0.2532, 0.2598)

    opt.name_net = 'deeplab'
    opt.batch_size = 8
    opt.num_workers = 2
    opt.epochs = 200
    opt.num_classes = 10
    # Folders
    opt.project_folder = '...' # main folder
    opt.train_data_folder = '...'
    opt.val_data_folder = '...'


    #----------------------------------------------------------------------------------------

    opt.results_folder = os.path.join(opt.project_folder, 'results')
    if not os.path.isdir(opt.results_folder):
        os.makedirs(opt.results_folder)

    return opt

# class = ["Rural", "Urban"]

# def buildDatasetPath(category, classType):
#     return getDirPath(category)+"/"+classType+"/"


# def buildCustomDataPath(pathUrl, classType):
#     return pathUrl+"/"+classType+"/"


def set_model(opt):
    if opt.name_net == 'unet':
        model = unets.U_Net(output_ch = opt.num_classes)
        
    if opt.name_net == 'r2unet':
        model = unets.R2U_Net(output_ch = opt.num_classes)
        
    if opt.name_net == 'attunet':
        model = unets.AttU_Net(output_ch = opt.num_classes)
        
    if opt.name_net == 'pspnet':
        model = smp.PSPNet('resnet34', in_channels=3, classes = opt.num_classes)
        
    if opt.name_net == 'pretrained_pspnet':
        model = smp.PSPNet(
            encoder_name='resnext50_32x4d', 
            in_channels=3,
            encoder_weights='imagenet', 
            classes=opt.num_classes, 
            activation='softmax2d')

    if opt.name_net == 'deeplab':
        model = deeplabv3_resnet50(num_classes = opt.num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion, optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def set_loader(opt):

    # normalize = transforms.Normalize(mean=opt.mean, std=opt.std)

    data_transforms = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.Resize((128, 256)),
    ])

    train_dataset = SegmentationDataset(opt.train_data_folder, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataset = SegmentationDataset(opt.val_data_folder)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    return train_loader, val_loader


def test(model, test_loader, criterion, opt):
    model.eval()
    val_loss = 0
    val_dice = 0
    total_num = 0
    with torch.no_grad():
        for image, mask in test_loader:
            if torch.cuda.is_available():
                image = image.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                
            if opt.name_net == 'deeplab':
                output = model(image)['out'] #with deeplab from torch
            else:
                output = model(image)
                
            mask = torch.squeeze(mask, dim = 1)
            loss = criterion(output, mask.long()) # mask.long() if using cross entropy
            total_num += mask.shape[0]
            val_loss += loss.item() * mask.shape[0]
            dice = metrics.new_dice_coef(output.argmax(dim=1), mask)
            val_dice += dice
    
    val_loss = val_loss / total_num      
    val_dice = val_dice / len(test_loader)
    print("validation loss", val_loss)
    print("validation DICE coefficient", val_dice) 
    
    return val_loss, val_dice



def train(model, train_loader, test_loader, criterion, optimizer, epoch, opt):
    model.train()
    total_loss, total_num = 0.0, 0
    idx = 0
    # for image, mask in train_loader:
    for image, mask in train_loader:
        optimizer.zero_grad()
        # print(torch.unique(mask)) # Must be integers
        if torch.cuda.is_available():
            image = image.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

        # forward + backward + optimize
        if opt.name_net == 'deeplab':
            output = model(image)['out'] #with deeplab from torch
        else:
            output = model(image)
        
        mask = torch.squeeze(mask, dim = 1)
        # print('mask shape', mask.shape)
        # print('output shape', output.shape)
        loss = criterion(output, mask.long()) # mask.long() if using cross entropy
        loss.backward()
        optimizer.step()

        total_num += mask.shape[0]
        total_loss += loss.item() * mask.shape[0]

        if (idx + 1) % opt.print_freq == 0:
            print('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {}'.format(epoch, opt.epochs,
                                                                    optimizer.param_groups[0]['lr'],
                                                                    total_loss / total_num))

            sys.stdout.flush()
        idx += 1

        gc.collect()
        torch.cuda.empty_cache()
    val_loss, val_dice = test(model, test_loader, criterion, opt)
    print("train() function - epoch total_loss", total_loss / total_num)
    train_loss = total_loss / total_num
    return train_loss, val_loss, val_dice 


def main():
    torch.cuda.empty_cache()
    opt = parse_option()
    print(opt)
    # build data loader
    train_loader, test_loader = set_loader(opt)
    print('train e test data loader created...')

    # define model
    model, criterion, optimizer = set_model(opt)
    save_file_1 = os.path.join(opt.results_folder, (str(opt.name_net) + '_best.pth'))

    # TRAINING
    print("Training...")

    # training routine
    best_val_dice = 0
    train_loss_values, val_loss_values, val_dices = [], [], []

    for epoch in range(1, opt.epochs + 1):

        train_loss, val_loss, val_dice = train(model, train_loader, test_loader, criterion, optimizer, epoch, opt)
        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)
        val_dices.append(val_dice)
        
        # save best model
        if val_dice > best_val_dice:
            print("saving/updating current best model at epoch=" + str(epoch))
            save_model(model, optimizer, opt, epoch, save_file_1)
            best_val_dice = val_dice

        # save loss values and plot
        tloss_df = pd.DataFrame(train_loss_values)
        vloss_df = pd.DataFrame(val_loss_values)
        dice_df = pd.DataFrame(val_dices)
        tloss_df.to_csv(opt.results_folder +'/' + (str(opt.name_net) + '_train_loss.csv'))
        vloss_df.to_csv(opt.results_folder +'/' + (str(opt.name_net) + '_val_loss.csv'))
        dice_df.to_csv(opt.results_folder +'/' + (str(opt.name_net) + '_val_dice.csv'))
        
        plt.figure(figsize=(15, 10))
        plt.plot(train_loss_values, label = 'train loss')
        plt.ylabel('train loss value')
        plt.xlabel('epochs')
        plt.savefig(opt.results_folder +'/' + str(opt.name_net) + ' _train_loss.png')
        plt.close()
        
        plt.figure(figsize=(15, 10))
        plt.plot(val_loss_values, label = 'validation loss')
        plt.ylabel('validation loss value')
        plt.xlabel('epochs')
        plt.savefig(opt.results_folder +'/' + str(opt.name_net) + ' _val_loss.png')
        plt.close()
        
        plt.figure(figsize=(15, 10))
        plt.plot(val_dices)
        plt.ylabel('dice value')
        plt.xlabel('epochs')
        plt.savefig(opt.results_folder +'/' + str(opt.name_net) + ' _val_dice.png')
        plt.close()


    # save the last model
    save_file_2 = os.path.join(opt.results_folder, (str(opt.name_net) + '_last.pth'))
    save_model(model, optimizer, opt, opt.epochs, save_file_2)
    

if __name__ == '__main__':
    main()
