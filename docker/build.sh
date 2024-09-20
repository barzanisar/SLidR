#!/bin/bash

docker build -t slidr -f Dockerfile ..
#docker build -t slidr:torch230 -f Dockerfile_new ..

# sudo SINGULARITY_NOHTTPS=1 singularity build slidr.sif docker-daemon://slidr:latest
#sudo SINGULARITY_NOHTTPS=1 singularity build slidr_torch230.sif docker-daemon://slidr:torch230

# Commit changes to a docker img:
# sudo docker images
# # check the image id
# sudo docker run -it <image_id> bash
# E.g:
# sudo docker run -it cf0f3ca922e0 bash
# <modify image>
# exit
# sudo docker ps -a
# sudo docker commit [CONTAINER_ID] [new_image_name]


# Push docker image to docker hub
#create a repo in docker hub called barzanisar/also
#docker tag slidr:latest barzanisar/slidr:latest
#docker push barzanisar/slidr:latest


#in turing (not enough memory error on lovelace)
#docker pull barzanisar/slidr:latest
#docker images

# Or try (doesnt work)
#singularity pull also.sif docker://barzanisar/also:latest