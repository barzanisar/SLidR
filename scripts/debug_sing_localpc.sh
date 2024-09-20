#!/bin/bash

# die function
die() { echo "$*" 1>&2 ; exit 1; }

NUSCENES=$(readlink -f ./datasets/nuscenes)
SUPERPIXELS=$(readlink -f ./superpixels)
echo "$NUSCENES"
echo "$SUPERPIXELS"

NUSCENES_VOLUME=$NUSCENES:/slidr/datasets/nuscenes
SUPERPIXELS_VOLUME=$SUPERPIXELS:/slidr/superpixels

CUR_DIR=$(pwd)
PROJ_DIR=$(dirname $CUR_DIR) #also_selfsup
echo "$CUR_DIR"
echo "$PROJ_DIR"


# Variables accessible by command line
CFG=slidr_minkunet.yaml
PRETRAIN_EXTRA_TAG=try_0
CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1

# Paths
SING_IMG=/home/barza/SLidR/slidr.sif

# Get command line arguments
while :; do
    case $1 in
    -h|-\?|--help)
        show_help    # Display a usage synopsis.
        exit
        ;;
    # train.py parameters
    -c|--cfg)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CFG=$2
            shift
        else
            die 'ERROR: "--cfg" requires a non-empty option argument.'
        fi
        ;;
    -s|--sing)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SING_IMG=/raid/home/nisarbar/singularity/$2.sif
            shift
        else
            die 'ERROR: "--sing" requires a non-empty option argument.'
        fi
        ;;
    -p|--pretrain_extra_tag)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAIN_EXTRA_TAG=$2
            shift
        else
            die 'ERROR: "--pretrain_extra_tag" requires a non-empty option argument.'
        fi
        ;;
    -u|--cuda_visible_devices)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CUDA_VISIBLE_DEVICES=$2
            NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
            shift
        else
            die 'ERROR: "--cuda_visible_devices" requires a non-empty option argument.'
        fi
        ;;
    # Additional parameters
    -?*)
        printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
        ;;
    *)               # Default case: No more options, so break out of the loop.
        break
    esac

    shift
done

PROJ_DIR=$(pwd)
echo "Proj_dir=$PROJ_DIR"


BASE_CMD="SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY
SINGULARITYENV_NCCL_BLOCKING_WAIT=1
singularity exec
--nv
--pwd /slidr
--bind $PROJ_DIR:/slidr \
--bind $NUSCENES_VOLUME \
--bind $SUPERPIXELS_VOLUME \
$SING_IMG \
bash
"
echo $BASE_CMD
eval $BASE_CMD


# NOTE: this singularity env takes PATH from bashrc so export PATH from turing's singularity env
# Do this in the singularity shell
#export PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin


