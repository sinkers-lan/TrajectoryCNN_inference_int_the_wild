#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
cd ../..
savepath='results/h36m/v2'
modelpath='checkpoints/h36m/v2'
pretrain_modelpath='checkpoints/h36m/v1/model.ckpt-769500'
realtestfile='data/h36m20/my_test/outputfile.npy'
#sleep 4h
logname='logs/h36m/v2_test.log'
nohup python -u train_TrajectoryCNN_h36m.py \
    --is_training False \
    --dataset_name skeleton \
    --train_data_paths data/h36m20/h36m20_train_3d.npy \
    --valid_data_paths data/h36m20/h36m20_val_3d.npy \
    --test_data_paths data/h36m20/testset \
    --real_test_file ${realtestfile} \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --bak_dir ${bak_path}   \
    --pretrained_model ${pretrain_modelpath} \
    --input_length 10 \
    --seq_length 20 \
    --stacklength 4 \
    --filter_size 3 \
    --lr 0.0001 \
    --batch_size 16 \
    --sampling_stop_iter 0 \
    --max_iterations 3000000 \
    --display_interval 10 \
    --test_interval 500 \
    --n_gpu 2 \
    --snapshot_interval 500  >>${logname}  2>&1 &

tail -f ${logname}
#--pretrained_model checkpoints/ske_predcnn/model.ckpt-1000 \
# --pretrained_model ${pretrain_modelpath}  \



