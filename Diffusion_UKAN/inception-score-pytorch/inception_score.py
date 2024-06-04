import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import os
from skimage import io
import cv2
import os
import numpy as np
from scipy.stats import entropy
import torchvision.datasets as dset
import torchvision.transforms as transforms

import argparse
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=32):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.image_files = os.listdir(folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.folder, image_file)
        image = io.imread(image_path)
  
        if self.transform:
            image = self.transform(image)
        return image
    
class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


if __name__ == '__main__':

    # cifar = dset.CIFAR10(root='data/', download=True,
    #                          transform=transforms.Compose([
    #                              transforms.Resize(32),``
    #                              transforms.ToTensor(),
    #                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                          ])
    # )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/data/wyli/code/TinyDDPM/Output/unet_busi/Gens/')

    args = parser.parse_args()

    dataset = UnlabeledDataset(args.data_root, transform=transform)
    
    print ("Calculating Inception Score...")
    print (inception_score(dataset, cuda=True, batch_size=1, resize=True, splits=10))


