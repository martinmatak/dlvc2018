#!/bin/bash
#PBS -S /bin/bash
#PBS -m bea
#PBS -M aliamini.r@gmail.com

# Request free GPU
export CUDA_VISIBLE_DEVICES=$(getFreeGPU)
python3 /home/e1227507/group15/submissions/assignment1/AliA/dlvc2018/assignment2/cnn_cats_dogs_pt3.py
