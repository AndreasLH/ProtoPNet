#!/bin/sh
#BSUB -J augment
#BSUB -o augment%J.out
#BSUB -e augment%J.err
#BSUB -n 10
#BSUB -R "rusage[mem=8G]"
#BSUB -W 10:00
#BSUB -N
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load python3/3.10.7

# load CUDA (for GPU support)
# module load cuda/11.7

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source torch/bin/activate

python img_aug.py

