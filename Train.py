import argparse
import csv
import os
import torch
import time
import numpy as np
import glob
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")
import PIL.Image as Image
from torch.utils.data import Dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import explained_variance_score

from model import MSAGHNet

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataset(root, mode):
    imgs = []
    img_root1 = root + '/img/*.*'
    mask_root1 = root + '/mask/*.*'
    img_data = glob.glob(img_root1)
    mask_label = glob.glob(mask_root1)
    for i in range(len(img_data)):
        img = os.path.join(img_data[i])
        mask = os.path.join(mask_label[i])
        imgs.append((img, mask))
    return imgs

class SetDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, mode="Train"):
        imgs = make_dataset(root, mode)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path).convert('RGB')
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y[0, :, :]

    def __len__(self):
        return len(self.imgs)


def dice_value(pred, target):
    num = pred.size(0)
    m1 = pred.contiguous().view(num, -1)
    m2 = target.view(num, -1)
    interesection = (m1 * m2).sum()
    return (2. * interesection) / (m1.sum() + m2.sum())



# def Dice(inp, target, eps=1):
#     input_flatten = inp.flatten()
#     target_flatten = target.flatten()
#     overlap = np.sum(input_flatten * target_flatten)
#     return np.clip(((2. * overlap) / (np.sum(target_flatten) + np.sum(input_flatten) + eps)), 1e-4, 0.9999)

def dice_loss(pred, target):
    smooth = 1
    num = pred.size(0)
    m1 = pred.contiguous().view(num, -1)
    m2 = target.view(num, -1)
    interesection = (m1 * m2).sum()
    return 1 - ((2. * interesection + smooth) / (m1.sum() + m2.sum() + smooth))

def Tensor_Resize(input):
    scaled_2 = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
    scaled_4 = F.interpolate(input, scale_factor=0.25, mode='bilinear', align_corners=False)
    scaled_8 = F.interpolate(input, scale_factor=0.125, mode='bilinear', align_corners=False)
    scaled_16 = F.interpolate(input, scale_factor=0.0625, mode='bilinear', align_corners=False)
    return input, scaled_2, scaled_4, scaled_8, scaled_16

def multy_dice_loss(pred, target):

    loss1 = dice_loss(torch.sigmoid(pred[0]), target[0].unsqueeze(1))
    loss2 = dice_loss(torch.sigmoid(pred[1]), target[1].unsqueeze(1))
    loss3 = dice_loss(torch.sigmoid(pred[2]), target[2].unsqueeze(1))
    loss4 = dice_loss(torch.sigmoid(pred[3]), target[3].unsqueeze(1))
    loss5 = dice_loss(torch.sigmoid(pred[4]), target[4].unsqueeze(1))

    # loss = loss1*0.6 + loss2*0.1 + loss3*0.1 + loss4*0.1 + loss5*0.1  ## Loss1
    loss = loss1*0.30 + loss2*0.25 + loss3*0.20 + loss4*0.15 + loss5*0.10  ## Loss2
    # loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5
    # loss1, loss2, loss3, loss4, loss5
    return loss

def multy_bce_loss(pred, target):

    bce_loss = nn.BCEWithLogitsLoss()
    loss1 = bce_loss(torch.sigmoid(pred[0]), target[0])
    loss2 = bce_loss(torch.sigmoid(pred[1]), target[1])
    loss3 = bce_loss(torch.sigmoid(pred[2]), target[2])
    loss4 = bce_loss(torch.sigmoid(pred[3]), target[3])
    loss5 = bce_loss(torch.sigmoid(pred[4]), target[4])

    # loss = loss1*0.6 + loss2*0.1 + loss3*0.1 + loss4*0.1 + loss5*0.1   ## Loss1
    loss = loss1*0.30 + loss2*0.25 + loss3*0.20 + loss4*0.15 + loss5*0.10  ## Loss2
    # loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5
    # loss1, loss2, loss3, loss4, loss5
    return loss

def train_model(model, criterion, optimizer, dataload, val_data, num_epochs, used_model, datasets, save_model):
    dice_all = 0.0
    save_index = 5.0
    for epoch in range(1, num_epochs):
        val_step = 0
        for x, y in dataload:
            inputs = x.to(device)
            labels = y.to(device).unsqueeze(1)
            labels = Tensor_Resize(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            dice = dice_value(torch.sigmoid(outputs[0]), labels[0])
            dice_all += dice.item()
            dice_loss1 = multy_dice_loss(outputs, labels)
            BCE_loss = multy_bce_loss(outputs, labels)
            loss = dice_loss1 + BCE_loss
            loss.backward()
            optimizer.step()
        val_loss_all = 0.
        for x1, y1 in val_data:
            val_step += 1
            inputs1 = x1.to(device)
            labels1 = y1.to(device).squeeze()
            outputs1 = model(inputs1)[0].squeeze()
            outputs1 = torch.sigmoid(outputs1)
            val_loss1 = dice_loss(outputs1, labels1)
            val_loss2 = criterion(outputs1, labels1)
            val_loss = val_loss1 + val_loss2
            val_loss_all += val_loss.item()
        print("Validation Loss: [%s]" % (val_loss_all / val_step))
        if (val_loss_all / val_step) <= save_index:
            save_index = val_loss_all / val_step
            torch.save(model.state_dict(), '%s/%s_%s_model.pth' % (save_model, used_model, datasets))
            print("Save the best model")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

y_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512))
])

def train(args, used_model, data_nam, model):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    save_model = 'save_model/%s/%s' % (data_nam, used_model)
    os.makedirs(save_model, exist_ok=True)
    liver_dataset = SetDataset("./data/%s/train" % data_nam, transform=x_transforms,
                               target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataset = SetDataset("./data/%s/val" % data_nam, transform=x_transforms, target_transform=y_transforms,
                             mode="Validation")
    val_data = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    train_model(model, criterion, optimizer, dataloaders, val_data, args.num_epoch, used_model, data_nam,
                save_model)

if __name__ == '__main__':

    torch.cuda.set_device(0)

    parse = argparse.ArgumentParser()
    parse.add_argument("--epoch", type=int, default=0, help="the start of epoch")
    parse.add_argument("--num_epoch", type=int, default=10, help="the number of epoches")
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--dataset", type=str, default="PanSeg") ## "PanSeg", "BUSI", "Kavsir", "Sliver", "BKPS"
    parse.add_argument("--used_model", type=str, default="MSAGHNet")
    args = parse.parse_args()
    print(args)

    model = MSAGHNet(3, 1).to(device)
    train(args, args.used_model, args.dataset, model)