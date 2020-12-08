conda activate /home/jhy/env/miniconda3/envs/star
cd ..
python -W ignore imagenet_eval.py -a old_resnet50 --data /home/jhy/data/ --gpu-id 0 --resume ../pretrain/model_best.pth.tar -e 