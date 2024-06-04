from Diffusion.Train import train, eval, eval_tmp
import os
import argparse
import torch
def main(model_config = None):

    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
        modelConfig['batch_size'] = 64
        modelConfig['test_load_weight'] = 'ckpt_{}_.pt'.format(modelConfig['epoch'])
        for i in range(32):
            modelConfig["sampledImgName"] = "sampledImgName{}.png".format(i)
            eval(modelConfig)
    else:
        for i in range(1):
            modelConfig["sampledImgName"] = "sampledImgName{}.png".format(i)
            eval_tmp(modelConfig,1000) # for grid visualization
            # eval(modelConfig) # for metric evaluation

def seed_all(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(args.seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=str, default='eval')
    parser.add_argument('--dataset', type=str, default='cvc') # busi, glas, cvc
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--channel', type=int, default=64)
    parser.add_argument('--test_load_weight', type=str, default='ckpt_1000_.pt')
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=float, default=64) # 64 or 128
    parser.add_argument('--dataset_repeat', type=int, default=1) # didnot use
    parser.add_argument('--seed', type=int, default=0) 
    parser.add_argument('--model', type=str, default='UKan_Hybrid')
    parser.add_argument('--exp_nme', type=str, default='./')

    parser.add_argument('--save_root', type=str, default='released_models/ukan_cvc') 
    # parser.add_argument('--save_root', type=str, default='released_models/ukan_glas') 
    # parser.add_argument('--save_root', type=str, default='released_models/ukan_busi') 
    args = parser.parse_args()

    save_root = args.save_root
    if args.seed != 0:
        seed_all(args)

    modelConfig = {
        "dataset": args.dataset, 
        "state": args.state, # or eval
        "epoch": args.epoch,
        "batch_size": args.batch_size,
        "T": args.T,
        "channel": args.channel,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": args.num_res_blocks,
        "dropout": args.dropout,
        "lr": args.lr,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 64,
        "grad_clip": 1.,
        "device": "cuda", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": os.path.join(save_root, args.exp_nme, "Weights"),
        "sampled_dir": os.path.join(save_root, args.exp_nme, "FinalCheck"),
        "test_load_weight": args.test_load_weight,
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8,
        "model":args.model, 
        "version": 1,
        "dataset_repeat": args.dataset_repeat,
        "seed": args.seed,
        "save_root": args.save_root,
        }

    os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)
    os.makedirs(modelConfig["sampled_dir"], exist_ok=True)

    main(modelConfig)
