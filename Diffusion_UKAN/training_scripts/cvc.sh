##!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

conda activate kan

GPU=0
MODEL='UKan_Hybrid'
EXP_NME='UKan_cvc'
SAVE_ROOT='./Output/'
DATASET='cvc'

cd ../

CUDA_VISIBLE_DEVICES=${GPU} python Main.py \
--model ${MODEL} \
--exp_nme ${EXP_NME}  \
--batch_size 32  \
--channel 64 \
--dataset ${DATASET} \
--epoch 1000 \
--save_root ${SAVE_ROOT} 
# --lr 1e-4 

# calcuate FID and IS
CUDA_VISIBLE_DEVICES=${GPU} python -m pytorch_fid "data/${DATASET}/images_64/" "${SAVE_ROOT}/${EXP_NME}/Gens" > "${SAVE_ROOT}/${EXP_NME}/FID.txt" 2>&1

cd inception-score-pytorch

CUDA_VISIBLE_DEVICES=${GPU} python inception_score.py --data-root "${SAVE_ROOT}/${EXP_NME}/Gens"  > "${SAVE_ROOT}/${EXP_NME}/IS.txt" 2>&1
