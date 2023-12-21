### FloodNet segmentation, semi-self supervised training, pretrain on LoveDA.
Remeber to set correct paths to folders in the two main!  

main_loveda.py is for pretrain on grouped LoveDA, main_pseudo_labels.py is for training on FloodNet  

DATA (LoveDA grouped and FloodNet) can be downloaded here: https://drive.google.com/drive/folders/1x9-NhDO5h_4ckUpCDwcRY_6F9tV98c-K?usp=drive_link  

Data is organized in a different way in the kaggle notebooks. To execute scripts use data linked above.

Note that FloodNetDataset_pseudolabels.py, FloodNetDataset.py, and segmentation_dataset.py are used to load data for FloodNet labeled, FloodNet unlabeled, and LoveDA (labeled)  

Notebooks (can be visualized on Kaggle) are examples with less epochs.

