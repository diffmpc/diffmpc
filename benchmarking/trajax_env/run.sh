#!/bin/bash
docker run --rm -it \
 -v "$PWD"/../..:/workspace \
 -v "$PWD"/start.sh:/start.sh \
 -e DISPLAY=$DISPLAY \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 --gpus all \
 trajax bash -c "/start.sh bash"
