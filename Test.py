import argparse
import os
import cv2
import torch
import warnings
import numpy as np
import glob
from medpy import metric
import PIL.Image as Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import MSAGHNet
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataset(root):
    imgs = []
    img_path = root + '/img/*.*'
    mask_path = root + '/mask/*.*'
    data_nam = glob.glob(img_path)
    mask_data = glob.glob(mask_path)
    for i in range(len(data_nam)):
        img = os.path.join(data_nam[i])
        mask = os.path.join(mask_data[i])
        imgs.append((img, mask))
    return imgs

class GetDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
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
        return img_x, img_y[0,:,:], x_path

    def __len__(self):
        return len(self.imgs)

smooth = 0.00001


def calculate_metric_percase(pred, gt):
    pred[pred > 0.5] = 1
    pred[pred != 1] = 0
    gt[gt > 0] = 1
    try:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        hd = metric.binary.hd(pred, gt)
        return dice, hd95, jc, hd
    except:
        return 0, 0, 0, 0

def compute(dataset, save_path):
    path1 = r'./%s' % save_path
    path2 = r'./data/%s/test/mask' % dataset
    list1 = os.listdir(path1)
    list2 = os.listdir(path2)
    L = len(list1)
    piex = 127
    Dice_means = 0.
    IOU_means = 0.
    HD_means = 0.
    for i, j in enumerate(list2):
        img1 = cv2.imread(os.path.join(path1, list1[i]), 0)
        img2 = cv2.imread(os.path.join(path2, list2[i]), 0)
        img2 = cv2.resize(img2, (512, 512), interpolation=cv2.INTER_NEAREST)
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        img11 = np.where(img1 >= piex, 1, 0)
        img22 = np.where(img2 >= piex, 1, 0)
        dice_num = calculate_metric_percase(img11, img22)[0]
        Dice_means += dice_num
        hd95_num = calculate_metric_percase(img11, img22)[1]
        HD_means += hd95_num
        iou_num = calculate_metric_percase(img11, img22)[2]
        IOU_means += iou_num

    rec = "IOU:%0.6f, DICE:%0.6f, HD95:%0.6f" % (
        IOU_means / L, Dice_means / L, (HD_means / L))
    print(rec)

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

y_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512))
])


def test_model(model, dataload, dataset, save_load, used_model):
    for x, y, name in dataload:
        inputs = x.to(device)
        output,_,_,_,_ = model(inputs)
        outputs = output
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        outputs = torch.cat([output] * 3, dim=1)[0, :, :, :]
        img_y = torch.squeeze(outputs).detach().cpu().numpy()
        img_y = img_y.transpose(1, 2, 0)
        plt.imsave('%s/%s.png' % (save_load, name[0].split("\\")[-1].split(".")[0]), img_y)
    compute(dataset, save_load)


def test(used_model, data_nam, model):
    test_dataset = GetDataset("data/%s/test" % data_nam, transform=x_transforms, target_transform=y_transforms)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    model.load_state_dict(
            torch.load(
                'save_model/%s/%s/%s_%s_model.pth' % (data_nam, used_model, used_model, data_nam),
                map_location='cpu'))
    save_load = "output/%s/%s" % (data_nam, used_model)
    os.makedirs(save_load, exist_ok=True)
    test_model(model, test_data, data_nam, save_load, used_model)


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--num_epoch", type=int, default=100, help="the number of epoches")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--dataset", type=str, default="PanSeg") ## "PanSeg", "BUSI", "Kavsir", "Sliver", "BKPS"
    parse.add_argument("--used_model", type=str, default="MSAGHNet")
    args = parse.parse_args()
    print(args)

    model = MSAGHNet(3, 1).to(device)
    test(args.used_model, args.dataset, model)
