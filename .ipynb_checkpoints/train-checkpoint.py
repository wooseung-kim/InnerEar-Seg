import torch
import os
import glob
import time
import random
import argparse
import numpy as np
import monai
import nibabel as nib
from tqdm import tqdm
from monai.transforms import *
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import (
    DataLoader,
    Dataset,
    CacheDataset,
    decollate_batch,
)
from labelPreservingDataAug import RandZoomInverseZoomd, RandRotateInverseRotated

torch.backends.cudnn.benchmark = True

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed) 
    print(f"Random seed set as {seed}")

set_seed(42)

parser = argparse.ArgumentParser(
     description="train code"
)
parser.add_argument(
    "--train-dir",
    type=str,
    required=False,
    default='/mnt/earmri/ws/subRegionSeg/data1/',
    help="Path to train image directory"
)
parser.add_argument(
    "--valid-dir",
    type=str,
    required=False,
    default='/mnt/earmri/data_ts/',
    help="Path to validation image directory"
)
parser.add_argument(
    "--label-dir",
    type=str,
    required=False,
    default='/mnt/earmri/ws/subRegionSeg/data2/',
    help="Path to label directory"
)
parser.add_argument(
    "--log-dir",
    type=str,
    default='./log_dir',
    help="Path to log directory"
)

args = parser.parse_args()
train_dir = args.train_dir
valid_dir = args.valid_dir
label_dir = args.label_dir
log_dir = args.log_dir

log_dir = os.path.normpath(log_dir)
if os.path.exists(log_dir)==False:
    os.mkdir(log_dir)

data_Root_tr = os.path.normpath(train_dir)
data_Root_tst_t2 = os.path.normpath(valid_dir)
data_Root_tst_lbl = os.path.normpath(label_dir)

plist_tr = []
exclude = [3,16,46,48]
for i in range(1,55):
    if i in exclude: continue
    if os.path.exists(os.path.join(data_Root_tr, f"{i}")):
        plist_tr.append(i)

plist_tst_t2 = []
for i in range(55,79):
    if os.path.exists(os.path.join(data_Root_tst_t2, f"{i}")):
        plist_tst_t2.append(i)

plist_tst_lbl = []
for i in range(55,79):
    if os.path.exists(os.path.join(data_Root_tst_lbl, f"{i}")):
        plist_tst_lbl.append(i)

train_idx = np.arange(0,len(plist_tr))
train_data_dicts = [
    {
        "og": os.path.join(data_Root_tr,str(plist_tr[idx]),"t2gd_reg.nii.gz"),
        "image1": os.path.join(data_Root_tr,str(plist_tr[idx]),"t2gd_reg.nii.gz"),
        "seg": os.path.join(data_Root_tr,str(plist_tr[idx]),"IE_subRegionSeg_lblpreserve.nii.gz"),
    }
    for idx in train_idx
]
train_Data =  train_data_dicts

valid_idx = np.arange(0,len(plist_tst_t2))
valid_data_dicts = [
    {
        "image1": os.path.join(data_Root_tst_t2,str(plist_tst_t2[idx]),"t2gd.nii.gz"),
        "seg": os.path.join(data_Root_tst_lbl,str(plist_tst_lbl[idx]),"seg.nii.gz"),
    }
    for idx in valid_idx
]
val_Data = valid_data_dicts

print('# Train data:', len(train_Data))
print('# Valid data:', len(val_Data))

train_transforms = Compose(
    [
        LoadImaged(keys=["og","image1","seg"], image_only=False),
        EnsureChannelFirstd(keys=["og","image1","seg"]),
        ScaleIntensityd(keys=["og","image1"], minv=0.0, maxv=1.0),
        RandZoomInverseZoomd(
            keys=["image1"],
            prob = 0.75,
            min_zoom = 0.5,
            max_zoom = 2.0,
            mode = ['bilinear']
        ),
        RandGaussianNoised(keys=["image1"],
                           prob=.333, mean=0.0, std=0.0111, 
                           allow_missing_keys=False),
        RandGaussianSmoothd(keys=["image1"],
                            sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)
                            , approx='erf', prob=.333, allow_missing_keys=False),
        RandGaussianSharpend(keys=["image1"],
                             sigma1_x=(0.2,.4), sigma1_y=(0.2, .4), sigma1_z=(0.2, .4), 
                             sigma2_x=0.2, sigma2_y=0.2, sigma2_z=0.2, alpha=(5.0, 15.0)
                             , approx='erf', prob=.333, allow_missing_keys=False),
        RandCropByPosNegLabeld(
            keys=["og","image1","seg"],
            label_key="seg",
            spatial_size=(128, 128, 44),
            pos=1,
            neg=1,
            num_samples=2,
        ),
        RandRotateInverseRotated(keys=["image1"],
            mode=["bilinear"],
            range_x=0.75, range_y=0.0, range_z=0.0,
            prob=0.75),
        CenterSpatialCropd(keys=["og","image1","seg"],
                         roi_size=(96,96,32)),
        RandFlipd(
            keys=["og","image1","seg"],
            spatial_axis=[0],
            prob=0.5,
        ),
        RandFlipd(
            keys=["og","image1","seg"],
            spatial_axis=[1],
            prob=0.5,
        ),
        RandFlipd(
            keys=["og","image1","seg"],
            spatial_axis=[2],
            prob=0.5,
        ),
        RandRotate90d(
            keys=["og","image1","seg"],
            prob=0.25,
            max_k=3,
        ),
        ToTensord(keys=["og","image1","seg"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image1","seg"]),
        EnsureChannelFirstd(keys=["image1","seg"]),
        ScaleIntensityd(keys=["image1"], minv=0.0, maxv=1.0),
        RandCropByPosNegLabeld(
            keys=["image1","seg"],
            label_key="seg",
            spatial_size=(96, 96, 32),
            pos=1,
            neg=0,
            num_samples=2,
        ),
        ToTensord(keys=["image1","seg"]),
    ]
)

train_ds = CacheDataset(
    data = train_Data,
    transform=train_transforms,
)
val_ds = CacheDataset(
    data=val_Data,
    transform=val_transforms,
)

train_loader = DataLoader(
    train_ds, batch_size=2, shuffle=True,
)
valid_loader = DataLoader(
    val_ds, batch_size=1, shuffle=True, 
)

device= "cuda"
model = monai.networks.nets.SwinUNETR(in_channels=1, out_channels=4, use_v2=True).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

def validation(epoch_iterator_val):
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image1"]).to(device), batch["seg"].to(device)
            val_labels = torch.round(val_labels)
            
            # Merge left and right indexes
            val_labels[val_labels == 2] = 1
            val_labels[val_labels == 3] = 2
            val_labels[val_labels == 5] = 2
            val_labels[val_labels == 4] = 3
            val_labels[val_labels == 6] = 3
            val_labels[val_labels>3]=3
            val_labels[val_labels<0]=0
            
            val_outputs = model(val_inputs)
            val_outputs = val_outputs.softmax(1)
            
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (step, len(epoch_iterator_val), dice)
            )
                  
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        
        x, y = (batch["image1"]).to(device), batch["seg"].to(device)
        
        # Merge left and right indexes
        y=torch.round(y)
        y[y==2] = 1
        y[y==3] = 2
        y[y==5] = 2
        y[y==4] = 3
        y[y==6] = 3
        y[y>3] = 3
        y[y<0] = 0

        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                valid_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            metric_values.append(dice_val)
            
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(log_dir, "best_model.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
                file = open(f'{log_dir}/train.txt', 'a')
                file.write(f'global_step : {str(global_step)}\n')
                file.write('current dice best : ')
                file.write(str(dice_val_best)+'\n')
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                    
                )
        
        global_step += 1
        
        if global_step%100==0:        
            scheduler.step()
    
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)

    return global_step, dice_val_best, global_step_best


max_iterations = 4096
eval_num = 64
post_label = AsDiscrete(to_onehot=4)
post_pred = AsDiscrete(argmax=True, to_onehot=4)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                              lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=True)

while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )

torch.save(
    model.state_dict(), os.path.join(log_dir, f"last_model_step{global_step}.pth")
)
