# -------------------------------------------------------------------------
# Author:   Pietro Brugnolo
# Date:     20/11/2023
# Version:  1
# -------------------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import unets
import segmentation_models_pytorch as smp 
from torchvision.models.segmentation import deeplabv3_resnet50
from FloodNetDataset import FloodNetDataset
from sklearn.metrics import confusion_matrix
import seaborn as sn

def calculate_dice(df, y_pred, y_true):
    
    local_df = pd.DataFrame(columns = range(10))

    classes = np.unique(y_true.detach().cpu().numpy())
    for num_class in classes:
        target = (y_true == num_class)
        pred = (y_pred == num_class)
        intersect = (target * pred).sum()
        base = (target).sum() + (pred).sum()
        del(target); del(pred)
        score = (2 * intersect + 1e-6) / (base + 1e-6)
        local_df[num_class] = [score.item()]
        del(intersect); del(base)
    
    return  pd.concat([df, local_df], axis = 0)

# Floodnet labeled
mean = [-0.2417,  0.8531,  0.1789]
std = [0.9023, 1.1647, 1.3271] 

#---------------------------------------------------------------
data_folder = '' # TEST DATA
save_folder = '' # SAVE PATH

name_net = 'deeplab'
path_to_model = save_folder + '/' + name_net + '_best.pth'
num_classes = 10
batch_size = 1
h = 512
w = 512
compute_confusion_matrix = True
#---------------------------------------------------------------

if name_net == 'deeplab':
    model = unets.U_Net(output_ch = num_classes)
        
if name_net == 'r2unet':
    model = unets.R2U_Net(output_ch = num_classes)
        
if name_net == 'attunet':
    model = unets.AttU_Net(output_ch = num_classes)
        
if name_net == 'pspnet':
    model = smp.PSPNet('resnet34', in_channels=3, classes = num_classes)
        
if name_net == 'pretrained_pspnet':
    model = smp.PSPNet(
    encoder_name='resnext50_32x4d', 
    in_channels=3,
    encoder_weights='imagenet', 
    classes=num_classes, 
    activation='softmax2d')

if name_net == 'deeplab':
        model = deeplabv3_resnet50(num_classes = num_classes)  
    
transform = transforms.Compose([
        transforms.Resize((h,w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

transform_targ= transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((h,w)),
    ])

validation_dataset = FloodNetDataset(data_folder, transform=transform , target_transform=transform_targ)
test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    
checkpoint = (torch.load(path_to_model,map_location=torch.device('cpu')))
model.load_state_dict(checkpoint['model']) 
model.eval()

if compute_confusion_matrix:
    y_pred = []
    y_true = []
    
df = pd.DataFrame(columns = range(10))
with torch.no_grad():
    for image, mask in test_loader:
        if torch.cuda.is_available():
            image = image.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            model = model.cuda()
                    
        if name_net == 'deeplab':
            output = model(image)['out'] #with deeplab from torch
        else:
            output = model(image)
        
        output = output.argmax(dim=1)
        
        #cheating a little to avoid the evaluation on class background
        background_map = (mask.squeeze(dim = 1) == 0) 
        output[background_map] = mask.squeeze(dim = 1).long()[background_map]
        
        classes = torch.unique(mask)
        df =  calculate_dice(df, mask, output)
        # print(df)
        if compute_confusion_matrix:
            y_pred.append(output.cpu().numpy())
            y_true.append(mask.cpu().numpy())
            
        

means = pd.DataFrame(df.mean(skipna=True))
means = pd.concat([means,means.mean(skipna=True)], axis = 0)
means.to_csv(save_folder + '/dice_scores_' + name_net + '.csv')

if compute_confusion_matrix:
    
    class_list = (
        "background",
        "building \n flooded",
        "building \n non flooded",
        "road \n flooded",
        "road \n non flooded",
        "water",
        "tree",
        "vehicle",
        "pool",
        "grass",
    )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = np.ndarray.flatten(y_true)
    y_pred = np.ndarray.flatten(y_pred)

    # Exclude the "background" class from the confusion matrix
    class_list_filtered = [cls for cls in class_list if cls != "background"]

    # Get the indices corresponding to the filtered class list
    indices_filtered = [class_list.index(cls) for cls in class_list_filtered]

    cf_matrix = confusion_matrix(y_true, y_pred)

    # Use the filtered indices for both rows and columns
    df_cm = pd.DataFrame(
        cf_matrix[np.ix_(indices_filtered, indices_filtered)] / cf_matrix[np.ix_(indices_filtered, indices_filtered)].sum(axis=1)[:, None],
        index=[i for i in class_list_filtered],
        columns=[i for i in class_list_filtered],
    )

    plt.figure(figsize=(7.5, 7))
    plt.subplots_adjust(left=0.15, bottom=0.21, right=0.998, top=0.936)
    sn.heatmap(df_cm, annot=True, cmap="Blues", annot_kws={"fontsize": 10})
    plt.savefig(save_folder + '/confusion_matrix' + name_net + '.png')
    plt.show()
