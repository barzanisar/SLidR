#!/bin/bash

NUSCENES=$(readlink -f ../datasets/nuscenes)
SUPERPIXELS=$(readlink -f ../superpixels)
echo "$NUSCENES"
echo "$SUPERPIXELS"

NUSCENES_VOLUME=$NUSCENES:/slidr/datasets/nuscenes
SUPERPIXELS_VOLUME=$SUPERPIXELS:/slidr/superpixels

CUR_DIR=$(pwd)
PROJ_DIR=$(dirname $CUR_DIR) #also_selfsup
echo "$CUR_DIR"
echo "$PROJ_DIR"


docker run -it \
        --runtime=nvidia \
        --net=host \
        --privileged=true \
        --ipc=host \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="$HOME/.$XAUTHORITY:/root/.Xauthority:rw" \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --env="WANDB_API_KEY=$WANDB_API_KEY" \
        --hostname="inside-DOCKER" \
        --name="slidr" \
        --volume $PROJ_DIR:/slidr \
        --volume ${NUSCENES_VOLUME} \
        --volume ${SUPERPIXELS_VOLUME} \
        --rm \
        slidr:torch230 bash
        # --rm \ #remove this if you want to keep the container to commit changes
        #slidr:latest is the original slidr env
        #slidr:torch230 is the ALSO env
        