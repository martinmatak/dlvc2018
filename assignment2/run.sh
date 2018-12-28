#!/bin/bash
#PBS -S /bin/bash
#PBS -m bea
#PBS -M martin.matak@gmail.com

# Request free GPU
export CUDA_VISIBLE_DEVICES=$(getFreeGPU)
python3 /home/e1635889/group15/submissions/assignment1/martin/dlvc2018/assignment2/cnn_cats_dogs.py
