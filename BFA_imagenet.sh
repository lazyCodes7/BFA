#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"alpha")
    PYTHON="/home/elliot/anaconda3/envs/pytorch041/bin/python" # python environment path
    TENSORBOARD='/home/elliot/anaconda3/envs/pytorch041/bin/tensorboard' # tensorboard environment path
    data_path='/home/elliot/data/imagenet' # dataset path
    ;;
esac

DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir /"content"/${DATE}/
fi

############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=vit_image_classification
dataset=imagenet
test_batch_size=64
data_path=/content/imagenet-mini

attack_sample_size=64 # number of image samples used for BFA
n_iter=50 # maximum allowed PBS iterations
k_top=10 # only check k_top weights with top gradient ranking in each layer

save_path=/"content"/${DATE}/${dataset}_${model}_BFA
tb_path=${save_path}/tb_log  #tensorboard log path

############### Neural network ############################
{
python /content/BFA/main.py --dataset ${dataset} \
    --arch ${model} --data_path ${data_path}   \
    --test_batch_size ${test_batch_size} --workers 8 --ngpu 1 --gpu_id 1 \
    --print_freq 50 \
    --bfa \
    --reset_weight \
    --n_iter ${n_iter} --k_top ${k_top} \
    --attack_sample_size ${attack_sample_size} \
    # --random_bfa
} &
############## Tensorboard logging ##########################
{
if [ "$enable_tb_display" = true ]; then 
    sleep 30 
    wait
    $TENSORBOARD --logdir $tb_path  --port=6006
fi
} &
{
if [ "$enable_tb_display" = true ]; then
    sleep 45
    wait
    case $HOST in
    "Hydrogen")
        firefox http://0.0.0.0:6006/
        ;;
    "alpha")
        google-chrome http://0.0.0.0:6006/
        ;;
    esac
fi 
} &
wait