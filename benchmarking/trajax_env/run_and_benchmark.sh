#!/bin/bash
docker run --rm -it \
 -v "$PWD"/../..:/workspace \
 -v "$PWD"/start_benchmark.sh:/start_benchmark.sh \
 -e DISPLAY=$DISPLAY \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 --gpus all \
 trajax bash -c "/start_benchmark.sh bash"
