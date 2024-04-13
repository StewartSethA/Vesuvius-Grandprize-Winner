import os.path as osp
import os
os.environ["WANDB_MODE"] = "offline"
import torch
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import random
import yaml

import numpy as np
import pandas as pd

import wandb

from torch.utils.data import DataLoader

import pandas as pd
import os
import random
from contextlib import contextmanager
import cv2

import scipy as sp
import numpy as np
import fastnumpyio as fnp
import pandas as pd

from tqdm.auto import tqdm

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW

import datetime
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from i3dallnl import InceptionI3d
#from pygoi3d_simple import InceptionI3d
import torch.nn as nn
import torch
from warmup_scheduler import GradualWarmupScheduler
from scipy import ndimage
import time
import json
import numba
from numba import jit

class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = './' #'/content/gdrive/MyDrive/vesuvius_model/training'
    comp_folder_name = './' #'/content/gdrive/MyDrive/vesuvius_model/training'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = './' #f'/content/gdrive/MyDrive/vesuvius_model/training'
    basepath = "/media/seth/WDC-4x1TB-Raid-10GBps/Vesuvius/"
    exp_name = 'pretraining_all'

    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = 'Unet'
    # backbone = 'efficientnet-b0'
    # backbone = 'se_resnext50_32x4d'
    backbone='resnet3d'
    in_chans = 30 # 65
    encoder_depth=5
    # ============== training cfg =============
    start_idx = 15
    end_idx = 45
    scale = 2
    size = 64
    tile_size = 256
    stride = tile_size // 8

    train_batch_size = 384 # 32
    valid_batch_size = train_batch_size
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 30 # 30

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    # lr = 1e-4 / warmup_factor
    lr = 2e-5
    # ============== fold =============
    #valid_id = '20230820203112'
    valid_id = '20231210132040'

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== fixed =============
    pretrained = True
    inf_weight = 'best'  # 'best'

    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 100

    print_freq = 50
    num_workers = 8

    seed = 0

    # ============== set dataset path =============
    print('set dataset path')

    outputs_path = f'./outputs' #/content/gdrive/MyDrive/vesuvius_model/training/outputs'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + \
        f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'

    # ============== augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.6),

        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.1,p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2), 
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
    rotate = A.Compose([A.Rotate(8,p=1)])
def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False
def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)
def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    # set_env_name()
    # set_dataset_path(cfg)

    if mode == 'train':
        make_dirs(cfg)
cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataloaders import *

# from resnetall import generate_model
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')
class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask



class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=256,enc='',with_norm=False,train_dataset=None,check_val_every_n_epoch=1):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters("size", "enc", "with_norm")
        self.hparams.pred_shape = pred_shape
        if isinstance(self.hparams.pred_shape, dict):
          self.mask_pred = {}
          self.mask_count = {}
          for k,pred_shape in self.hparams.pred_shape.items():
            self.mask_pred[k] = np.zeros(pred_shape)
            self.mask_count[k] = np.zeros(pred_shape)
        else:
          self.mask_pred = np.zeros(self.hparams.pred_shape)
          self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)

        self.backbone=InceptionI3d(in_channels=1,num_classes=512,non_local=True)        
        self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,20,256,256))], upscale=1)

        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)
        self.train_dataset = train_dataset


            
    def forward(self, x):
        if x.ndim==4:
            x=x[:,None]
        if self.hparams.with_norm:
            x=self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        
        return pred_mask
    
    def training_step(self, batch, batch_idx):
        x, y, xys, ids = batch
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        if torch.isnan(loss1):
            print("Loss nan encountered")
        self.log("train/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys,ids= batch
        batch_size = x.size(0)
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            x1,x2,y1,y2 = (int(c) for c in (x1,x2,y1,y2))
            #print(x1, x2, y1, y2, "id", ids[i], [(k,v.shape) for k,v in self.mask_pred.items()], [(k,v.shape) for k,v in self.mask_count.items()])
            self.mask_pred[ids[i]][y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[ids[i]][y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}
    
    def on_validation_epoch_end(self):
        if isinstance(self.hparams.pred_shape, dict):
          for k,pred_shape in self.hparams.pred_shape.items():
            self.mask_pred[k] = np.divide(self.mask_pred[k], self.mask_count[k], out=np.zeros_like(self.mask_pred[k]), where=self.mask_count[k]!=0)
            wandb_logger.log_image(key=f"masks_{k}", images=[np.clip(self.mask_pred[k],0,1)], caption=["probs"])
        else:
          self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
          wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])
        print("Done logging! Resetting mask...")
        #reset mask
        if isinstance(self.hparams.pred_shape, dict):
          self.mask_pred = {}
          self.mask_count = {}
          for k,pred_shape in self.hparams.pred_shape.items():
            self.mask_pred[k] = np.zeros(pred_shape)
            self.mask_count[k] = np.zeros(pred_shape)
        else:
          self.mask_pred = np.zeros(self.hparams.pred_shape)
          self.mask_count = np.zeros(self.hparams.pred_shape)
        print("Done resetting masks!")

        #self.train_dataset.labels = reload_masks(self.train_dataset.labels, CFG)
    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=CFG.lr)
    
        scheduler = get_scheduler(CFG, optimizer)
        return [optimizer],[scheduler]



class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 10, eta_min=1e-6)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler

#def scheduler_step(scheduler, avg_val_loss, epoch):
#    scheduler.step(epoch)
#   

torch.set_float32_matmul_precision('medium')

fragments=[CFG.valid_id] #'20230820203112']
scroll4_ids = ['20231111135340', '20231122192640', '20231210132040', '20240304141530', '20231215151901']
enc_i,enc,fold=0,'i3d',0
for fid in fragments:
    CFG.valid_id=fid
    fragment_id = CFG.valid_id
    run_slug=f'training_scrolls_valid={fragment_id}_{CFG.size}x{CFG.size}_submissionlabels_wild11'

    pred_shape = {}
    for scroll_id in scroll4_ids:
      valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{scroll_id}/{scroll_id}_inklabels.png", 0)
      if valid_mask_gt is None:
        valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{scroll_id}/{scroll_id}_mask.png", 0)
      pred_shape[scroll_id]=tuple(int(c/CFG.scale) for c in valid_mask_gt.shape)
    print("Got all validation scroll ID shapes for prediction and logging")
    norm=fold==1
    wandb_logger = WandbLogger(project="vesivus",name=run_slug+f'{enc}_finetune')

    print('FOLD : ',fold)
    multiplicative = lambda epoch: 0.9

    trainer = pl.Trainer(
        max_epochs=24,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        logger=wandb_logger,
        default_root_dir="./", #/content/gdrive/MyDrive/vesuvius_model/training/outputs",
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        precision='bf16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        strategy='ddp_find_unused_parameters_true',
        callbacks=[ModelCheckpoint(filename=f'wild12_64_{fid}_{fold}_fr_{enc}'+'{epoch}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=CFG.epochs),
                    ],
    )

    #train_ids = set(['20230702185753','20230929220926','20231005123336','20231012184423','20231007101619','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) - set([CFG.valid_id])
    train_ids = set(['20231210132040', '20230702185753','20230929220926','20231005123336','20231012184423','20231007101619','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) - set([CFG.valid_id])
    #train_ids = set(['20230702185753']) #,'20230929220926','20231005123336','20231012184423','20231007101619','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) - set([CFG.valid_id])
    #train_ids = set(['20231210132040']) #, '20230702185753','20230929220926','20231005123336','20231012184423','20231007101619','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) - set([CFG.valid_id])
    valid_ids = set(scroll4_ids) #set([CFG.valid_id])

    train_images, train_masks, train_xyxys, train_ids, valid_images, valid_masks, valid_xyxys, valid_ids = get_train_valid_dataset(CFG, train_ids, valid_ids, scale=CFG.scale)
    print(len(train_images))
    valid_xyxys = np.stack(valid_xyxys)
    train_dataset = CustomDataset(
        train_images, CFG, labels=train_masks, xyxys=train_xyxys, ids=train_ids, transform=get_transforms(data='train', cfg=CFG), scale=CFG.scale)
    valid_dataset = CustomDataset(
        valid_images, CFG,xyxys=valid_xyxys, labels=valid_masks, ids=valid_ids, transform=get_transforms(data='valid', cfg=CFG), scale=CFG.scale)

    train_loader = DataLoader(train_dataset,
                                batch_size=CFG.train_batch_size,
                                shuffle=True,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                                )
    valid_loader = DataLoader(valid_dataset,
                                batch_size=CFG.valid_batch_size,
                                shuffle=False,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True)

    model=RegressionPLModel(enc='i3d',pred_shape=pred_shape,size=CFG.size, train_dataset=train_dataset)

    wandb_logger.watch(model, log="all", log_freq=100)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    wandb.finish()
