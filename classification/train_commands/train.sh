conda activate /home/jhy/env/miniconda3/envs/star
cd ..
python -W ignore imagenet.py -a old_resnet50 --data /home/jhy/data/ --gpu-id 0 -c ../trained_results --epochs 300 --resume ../pretrain/old_resnet50.pth.tar -c ../trained_results  --schedule 50 100 150 --gamma 0.9 --lr 0.001