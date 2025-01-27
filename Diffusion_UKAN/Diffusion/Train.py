
import os
from typing import Dict
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, transforms
# from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.UNet import UNet, UNet_Baseline
from Diffusion.Model_ConvKan import UNet_ConvKan
from Diffusion.Model_UMLP import UMLP
from Diffusion.Model_UKAN_Hybrid import UKan_Hybrid
from Scheduler import GradualWarmupScheduler
from skimage import io
import os
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset
import sys


model_dict = {
    'UNet': UNet,
    'UNet_ConvKan': UNet_ConvKan, # dose not work
    'UMLP': UMLP,
    'UKan_Hybrid': UKan_Hybrid,
    'UNet_Baseline': UNet_Baseline,
}

class UnlabeledDataset(Dataset):
    def __init__(self, folder, transform=None, repeat_n=1):
        self.folder = folder
        self.transform = transform
        # self.image_files = os.listdir(folder) * repeat_n
        self.image_files = os.listdir(folder) 

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.folder, image_file)
        image = io.imread(image_path)
        if self.transform:
            image = self.transform(image)
        return image, torch.Tensor([0])


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    log_print = True
    if log_print:
        file = open(modelConfig["save_weight_dir"]+'log.txt', "w")
        sys.stdout = file
    transform = Compose([
        ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if modelConfig["dataset"] == 'cvc':
        dataset = UnlabeledDataset('data/cvc/images_64/', transform=transform, repeat_n=modelConfig["dataset_repeat"])
    elif modelConfig["dataset"] == 'glas':
        dataset = UnlabeledDataset('data/glas/images_64/', transform=transform, repeat_n=modelConfig["dataset_repeat"])
    elif modelConfig["dataset"] == 'glas_resize':
        dataset = UnlabeledDataset('data/glas/images_64_resize/', transform=transform, repeat_n=modelConfig["dataset_repeat"])
    elif modelConfig["dataset"] == 'busi':
        dataset = UnlabeledDataset('data/busi/images_64/', transform=transform, repeat_n=modelConfig["dataset_repeat"])
    else:
        raise ValueError('dataset not found')

    print('modelConfig: ')
    for key, value in modelConfig.items():
        print(key, ' : ', value)
        
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    
    print('Using {}'.format(modelConfig["model"]))
    # model setup
    net_model =model_dict[modelConfig["model"]](T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                    num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
        
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)

    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(1,modelConfig["epoch"]+1):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                # print version
                if log_print:
                    print("epoch: ", e, "loss: ", loss.item(), "img shape: ", x_0.shape, "LR: ", optimizer.state_dict()['param_groups'][0]["lr"])
        warmUpScheduler.step()
        if e % 50 ==0:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))
            modelConfig['test_load_weight'] = 'ckpt_{}_.pt'.format(e)
            eval_tmp(modelConfig, e)

    torch.save(net_model.state_dict(), os.path.join(
        modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))
    if log_print:
        file.close()
        sys.stdout = sys.__stdout__
    
def eval_tmp(modelConfig: Dict, nme: int):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = model_dict[modelConfig["model"]](T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
    
        model.load_state_dict(ckpt)
        
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
        # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        # save_image(saveNoisy, os.path.join(
            # modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]

        save_root = modelConfig["sampled_dir"].replace('Gens','Tmp')
        os.makedirs(save_root, exist_ok=True)
        save_image(sampledImgs, os.path.join(
            save_root,  modelConfig["sampledImgName"].replace('.png','_{}.png').format(nme)), nrow=modelConfig["nrow"])
        if nme < 0.95 * modelConfig["epoch"]:
            os.remove(os.path.join(
                modelConfig["save_weight_dir"], modelConfig["test_load_weight"]))

def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])

        model = model_dict[modelConfig["model"]](T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                    num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)

        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)     
        # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        # save_image(saveNoisy, os.path.join(
        #     modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]

        for i, image in enumerate(sampledImgs):
    
            save_image(image, os.path.join(modelConfig["sampled_dir"],  modelConfig["sampledImgName"].replace('.png','_{}.png').format(i)), nrow=modelConfig["nrow"])
