#!/bin/bash

docker build -t slidr -f Dockerfile ..
#docker build -t slidr:torch230 -f Dockerfile_new ..

sudo SINGULARITY_NOHTTPS=1 singularity build slidr.sif docker-daemon://slidr:latest
sudo SINGULARITY_NOHTTPS=1 singularity build slidr_torch230.sif docker-daemon://slidr:torch230