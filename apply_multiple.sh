#!/bin/bash
#SBATCH −−nodes=1
#SBATCH −−ntasks=1
#SBATCH −−gres=gpu:a40:1 
#SBATCH −−mem=16G
#SBATCH −−cpus−per-task=8
#SBATCH −−output=%j−%x . out

srun --pty --gres=gpu:a40:$1 --cpus-per-task=48 --mem=350G /bin/bash
