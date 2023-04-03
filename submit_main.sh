#!/bin/sh
#BSUB -J main
#BSUB -o main%J.out
#BSUB -e main%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 2
#BSUB -R "rusage[mem=8G]"
#BSUB -W 10:00
#BSUB -N
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load python3/3.10.7

# load CUDA (for GPU support)
module load cuda/11.7

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source torch/bin/activate

python main.py

