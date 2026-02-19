import torch
import os
import glob
import time
import random
import argparse
import numpy as np
import monai
import nibabel as nib
import scipy.ndimage as nd
import nibabel as nib

from tqdm import tqdm
from monai.transforms import *
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import (
    DataLoader,
    Dataset,
    CacheDataset,
)
from skimage import measure
from monai.inferers import sliding_window_inference

parser = argparse.ArgumentParser(
     description="test code"
)
parser.add_argument(
    "--test-dir",
    type=str,
    required=False,
    default='/mnt/earmri/ws/final/testset/paik',
    help="Path to test image directory"
)
parser.add_argument(
    "--log-dir",
    type=str,
    default='./log_dir',
    help="Path to log directory"
)

args = parser.parse_args()
test_dir = args.test_dir
log_dir = args.log_dir

device= "cuda"
model = monai.networks.nets.SwinUNETR(in_channels=1, out_channels=4, use_v2=True).to(device)
model.to(device)
model.load_state_dict(torch.load(f"./{log_dir}/best_model.pth"))

plist_test = glob.glob(os.path.join(test_dir, '*/t2gd.nii.gz'))
plist_test.sort()

test_ind = np.arange(0,len(plist_test))
data_dicts = [
    {
        "image1": plist_test[idx],
        "image2": plist_test[idx],
        "image3": plist_test[idx],
        "image4": plist_test[idx],
    }
    for idx in test_ind
]
test_Data = data_dicts
print('# Test data:', len(test_Data))

test_transforms = Compose(
    [
        LoadImaged(keys=["image1","image2","image3","image4"], image_only=False),
        EnsureChannelFirstd(keys=["image1","image2","image3","image4"]),
        ScaleIntensityd(keys=["image1","image2","image3","image4"], minv=0.0, maxv=1.0),
        Flipd(
            keys=["image2"],
            spatial_axis=0,
        ),
        Flipd(
            keys=["image3"],
            spatial_axis=1,
        ),
        Flipd(
            keys=["image4"],
            spatial_axis=2,
        ),
        ToTensord(keys=["image1","image2","image3","image4"]),
    ]
)
test_ds = Dataset(
    data=test_Data,
    transform=test_transforms,
)
test_loader = DataLoader(
    test_ds, batch_size=1, shuffle=False, 
)

model.eval()
with torch.no_grad():
    for step, batch in enumerate(tqdm(test_loader)):
        pid = batch["image1_meta_dict"]['filename_or_obj'][0].split('/')[-2]
        
        test_inputs = (batch["image1"]).to(device), (batch["image2"]).to(device), (batch["image3"]).to(device), (batch["image4"]).to(device)
        test_outputs = []
        for i, test_input in enumerate(test_inputs):
            test_output = sliding_window_inference(test_input, [96,96,32], 8, model, overlap=0.75, mode='gaussian')
            if i > 0:
                flip = monai.transforms.Flip(spatial_axis=i-1)
                test_output[0] = flip(test_output[0])
            test_outputs.append(test_output)

        res = torch.stack(test_outputs, axis=0)
        res = torch.mean(res, dim=0)

        res = res.softmax(1)

        cc=[]
        for ch in range(1, res.shape[1]): # except background channel
            r = res[0,ch].clone().detach().cpu().numpy()
            r[r<0.1]= 0
            earmask = np.zeros(r.shape).astype('int16')
            l, m = nd.label(r, structure=np.ones((3,3,3)))
            vols=[]
            for i in range(m+1):
                vols.append(np.sum(l==i))
            vind = np.argsort(vols)
            earmask[l==vind[-2]] = 1
            earmask[l==vind[-3]] = 1
            cc.append(earmask)
        masks = np.stack(cc, axis=0) # 4 channels [C,H,W,D]

        fg = res[0,1:].clone().detach().cpu().numpy()
        earmask = masks * fg # 4 channels [C,H,W,D] without background channel
        earmask = np.concatenate((res[0,[0]], earmask), axis=0) # concat background channel with foreground channels.
        earmask[0][(earmask[1]==0) & (earmask[2]==0) & (earmask[3]==0)] = 1.0
        
        earmask = earmask.argmax(0)
        
        dpath = os.path.join(os.path.dirname(batch["image1_meta_dict"]['filename_or_obj'][0]), 'dlouts')
        if not os.path.exists(dpath):
            os.makedirs(dpath)
        
        hinfo = nib.load(batch["image1_meta_dict"]['filename_or_obj'][0])
        h = nib.Nifti1Image(earmask.astype(np.int16), hinfo.affine, hinfo.header)
        nib.save(h,os.path.join(dpath, 'dlseg.nii.gz'))
