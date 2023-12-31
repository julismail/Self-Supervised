#import and add dataset
import os
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np
from lightly.data import LightlyDataset

num_workers = 8
batch_size = 512
seed = 1
max_epochs = 300
input_size = 128
num_ftrs = 32

pl.seed_everything(seed)

path_to_data = '/home/ismail/datasets/debi/Gray'

#data augmentation
transform = SimCLRTransform(input_size,vf_prob=0.5,rr_prob=0.5)

# torchvision transformation for embedding the dataset after training
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=utils.IMAGENET_NORMALIZE['mean'],
        std=utils.IMAGENET_NORMALIZE['std'],
    ),
])

dataset_train_simclr = LightlyDataset(
    input_dir=path_to_data, transform=transform
)

dataset_test = LightlyDataset(
    input_dir=path_to_data,
    transform=transforms
)

dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

from lightly.models.modules.heads import ProjectionHead
from lightly.loss import NTXentLoss


class SimCLRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = ProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, max_epochs
        )
        return [optim], [scheduler]
        
from pytorch_lightning.callbacks import TQDMProgressBar
gpus = 1 if torch.cuda.is_available() else 0

model = MalSSLModel()
trainer = pl.Trainer(
    max_epochs=max_epochs, accelerator='gpu', devices=1, 
                     callbacks=[TQDMProgressBar(refresh_rate=100)])

trainer.fit(model, dataloader_train_simclr)

#to save a model
pretrained_resnet_backbone = model.backbone

# to store the backbone and use it in another code
state_dict = {
    'resnet18_parameters': pretrained_resnet_backbone.state_dict()
}
torch.save(state_dict, 'model1.pth')

