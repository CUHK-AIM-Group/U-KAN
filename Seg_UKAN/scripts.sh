dataset=busi
input_size=256
python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
python val.py --name ${dataset}_UKAN 

dataset=glas
input_size=512
python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
python val.py --name ${dataset}_UKAN 

dataset=cvc
input_size=256
python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
python val.py --name ${dataset}_UKAN 







