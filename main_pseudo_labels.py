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
from FloodNetDataset import FloodNetDataset
from FloodNetDataset_pseudolabels import FloodNetDataset_pseudolabels
import metrics
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms as T
import unets
from PIL import Image 
from torchvision.models.segmentation import deeplabv3_resnet50
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import segmentation_models_pytorch as smp 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device, "\n")

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of training epochs')
    parser.add_argument('--project_folder', type=str, default=None,
                        help='project folder')
    parser.add_argument('--supervised_epochs', type=int, default=50,
                        help='number of epochs with training only on labeled data')
    
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
    parser.add_argument('--num_classes', type=int, default=10, help='path to image dataset')
    # # temperature
    # parser.add_argument('--temp', type=float, default=0.07,
    #                     help='temperature for loss function')
    
    parser.add_argument('--data_folder', type=str, default=None, help='path to image dataset')
    parser.add_argument('--resize_height', type=int, default=256, help='resize h')
    parser.add_argument('--resize_width', type=int, default=512, help='resize w')
    
    parser.add_argument('--b_factor', type=int, default=2, help=' unspu_b_size = sup_b_size * b_factor')
    parser.add_argument('--alpha', type=float, default=1.5, help=' unspu_b_size = sup_b_size * b_factor')
    parser.add_argument('--load_saved_model', type=bool, default=False, help='load model already trained')
    parser.add_argument('--path_to_pretrained_model', type=str, help='path to model already trained')
    parser.add_argument('--threshold_val_dice', type=int, help='val dice higher than threshold_val_dice -> model saved')
    
    opt = parser.parse_args()

    #----------------------------------------------------------------------------------------
    # Dataset
    opt.mean = [-0.2417,  0.8531,  0.1789]
    opt.std = [0.9023, 1.1647, 1.3271] 
    opt.name_net = 'unet' 
    opt.batch_size = 8
    opt.num_workers = 2
    opt.epochs = 200
    opt.num_classes = 10
    opt.resize_height = 512
    opt.resize_width = 512
    opt.supervised_epochs = 50
    opt.b_factor = 1
    opt.alpha = 1
    opt.load_saved_model = False
    opt.threshold_val_dice = 0.35
    # Folders
    opt.project_folder = '' # main folder
    opt.data_folder = '' # data folder, must lead to an folder that contains Train/, Val/, UnsupTrain/
    opt.path_to_pretrained_model = '' # if opt.load_saved_model == True seek here the model, must be in the form name_best.pth
    #----------------------------------------------------------------------------------------

    opt.results_folder = os.path.join(opt.project_folder, 'results_sample')
    if not os.path.isdir(opt.results_folder):
        os.makedirs(opt.results_folder)

    return opt

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
    
    if opt.name_net == 'pretrained_segformer':
        model = SegformerForSemanticSegmentation.from_pretrained(
            'nvidia/mit-b2',
            num_labels=opt.num_classes)
        
    if opt.name_net == 'segformer':
        configuration = SegformerConfig(num_labels = opt.num_classes)
        model = SegformerForSemanticSegmentation(configuration)
   
    criterion = nn.CrossEntropyLoss()
    criterion_psl = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum)
    optimizer = optim.AdamW(model.parameters(), lr=opt.learning_rate)
    # optimizer_psl = optim.AdamW(model.parameters(), lr=opt.learning_rate)
    if opt.load_saved_model:
        path = os.path.join(opt.path_to_pretrained_model, opt.name_net +'_best.pth')
        checkpoint = (torch.load(path,map_location=torch.device('cpu')))
        model.load_state_dict(checkpoint['model']) 
        
        
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion, criterion_psl, optimizer


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
    
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.Resize((opt.resize_height, opt.resize_width)),
        T.ToTensor(),
        T.Normalize(mean=opt.mean, std=opt.std)
    ])
    
    val_transform = T.Compose([
        T.Resize((opt.resize_height, opt.resize_width)),
        T.ToTensor(),
        T.Normalize(mean=opt.mean, std=opt.std)
    ])

    weak_transform = T.Compose([
        T.RandomHorizontalFlip(p = 0.5),
        T.RandomVerticalFlip(p = 0.5),
        T.Resize((opt.resize_height, opt.resize_width)),
        T.ToTensor(),
        T.Normalize(mean=opt.mean, std=opt.std),
    ])
    
    strong_transform = T.Compose([
        T.RandomHorizontalFlip(p = 0.5),
        T.RandomVerticalFlip(p = 0.5),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        T.RandomEqualize(),
        T.RandomAutocontrast(),
        T.RandomAdjustSharpness(sharpness_factor = 2),
        T.RandomPosterize(bits=2),
        T.Resize((256, 512)),
        T.ToTensor(),
    ])

    if opt.name_net == 'pretrained_segformer' or opt.name_net == 'segformer':
        train_target_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Resize((int(opt.resize_height/4), int(opt.resize_width/4))),
            T.PILToTensor(),
        ])
        
        val_target_transform = T.Compose([
        T.Resize((int(opt.resize_height/4), int(opt.resize_width/4))),
        T.PILToTensor(),
        ])
        
    else:
        train_target_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Resize((opt.resize_height, opt.resize_width)),
            T.PILToTensor(),
        ])
        
        val_target_transform = T.Compose([
        T.Resize((opt.resize_height, opt.resize_width)),
        T.PILToTensor(),
        ])

    train_dir = opt.data_folder + '/Train'
    val_dir = opt.data_folder + '/Val'
    unsup_train_dir = opt.data_folder +'/UnsupTrain'
    train_dataset = FloodNetDataset(train_dir, transform=train_transform , target_transform=train_target_transform)
    validation_dataset = FloodNetDataset(val_dir, transform=val_transform , target_transform=val_target_transform)
    unsup_train_dataset = FloodNetDataset_pseudolabels(unsup_train_dir, weak_transform=weak_transform, strong_transform = strong_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=opt.batch_size, shuffle=False)
    unsup_train_loader =  torch.utils.data.DataLoader(unsup_train_dataset, batch_size=opt.b_factor * opt.batch_size, shuffle=False)

    return train_loader, test_loader, unsup_train_loader


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
            elif opt.name_net == 'pretrained_segformer' or opt.name_net == 'segformer':
                output = model(image)
                output = output.logits
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



def train(model, train_loader, test_loader, unsup_train_loader, criterion, criterion_psl, optimizer, epoch, opt):
    model.train()
    total_loss, total_num = 0.0, 0
    idx = 0
    
    if epoch< opt.supervised_epochs:
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
            elif opt.name_net == 'pretrained_segformer' or opt.name_net == 'segformer':
                output = model(image)
                output = output.logits
            else:
                output = model(image)
                
            mask = torch.squeeze(mask, dim = 1)
            # print('mask shape', mask.shape)
            # print('output shape', output.shape)
            loss = (1/image.shape[0]) * criterion(output, mask.long()) # mask.long() if using cross entropy
            loss.backward()
            optimizer.step()
            total_num += mask.shape[0]
            total_loss += loss.item() * mask.shape[0]
        
        if (idx + 1) % opt.print_freq == 0:
            print('Fully_supervised-Train Epoch: [{}/{}], lr: {:.6f}, Loss: {}'.format(epoch, opt.epochs,
                                                                    optimizer.param_groups[0]['lr'],
                                                                    total_loss / total_num))
            sys.stdout.flush()
        idx += 1
    else:
        
        for (image, mask), (image_waug, image_saug) in zip(train_loader, unsup_train_loader):
            optimizer.zero_grad()
            # print(torch.unique(mask)) # Must be integers
            if torch.cuda.is_available():
                image = image.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                image_waug = image_waug.cuda(non_blocking=True)
                image_saug = image_saug.cuda(non_blocking=True)
            # forward + backward + optimize
            if opt.name_net == 'deeplab':
                output = model(image)['out'] #with deeplab from torch
                # pseudo_label = model(image_waug)['out']  #with deeplab from torch
                pseudo_output =  model(image_saug)['out'] #with deeplab from torch
            elif opt.name_net == 'pretrained_segformer' or opt.name_net == 'segformer':
                output = model(image)
                output = output.logits
                pseudo_output =  model(image_saug)
                pseudo_output = pseudo_output.logits
            else:
                output = model(image)
                # pseudo_label = model(image_waug)
                pseudo_output =  model(image_saug)
                
            mask = torch.squeeze(mask, dim = 1)
            # pseudo_label = pseudo_label.argmax(1)
            pseudo_label = pseudo_output.argmax(1)
            # print('pseudo_label shape', pseudo_label.shape)
            # print('pseudo_label unique', torch.unique(pseudo_label))
            # print('pseudo_output shape', pseudo_output.shape)
            # print('output shape', output.shape)
            cv_coeff = (epoch - opt.supervised_epochs)/(opt.epochs - opt.supervised_epochs)
            loss = (1/image.shape[0]) * criterion(output, mask.long()) + (1/image_waug.shape[0]) * cv_coeff * opt.alpha * criterion_psl(pseudo_output, pseudo_label.long()) # mask.long() if using cross entropy
            loss.backward()
            optimizer.step()
            total_num += mask.shape[0]
            total_loss += loss.item() * mask.shape[0]
            if (idx + 1) % opt.print_freq == 0:
                print('Self_supervised-Train Epoch: [{}/{}], lr: {:.6f}, Loss: {}'.format(epoch, opt.epochs,
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
    train_loader, test_loader, unsup_train_loader = set_loader(opt)
    print('train e test data loader created...')

    # define model
    model, criterion, criterion_psl, optimizer = set_model(opt)
    save_file_1 = os.path.join(opt.results_folder, (str(opt.name_net) + '_best.pth'))

    # TRAINING
    print("Training...")

    # training routine
    best_val_dice = opt.threshold_val_dice
    train_loss_values, val_loss_values, val_dices = [], [], []

    for epoch in range(1, opt.epochs + 1):

        train_loss, val_loss, val_dice = train(model, train_loader, test_loader, unsup_train_loader, criterion, criterion_psl, optimizer, epoch, opt)
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
