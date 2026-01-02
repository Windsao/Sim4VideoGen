#!/bin/bash
#SBATCH -A p32958 ## Required: your allocation/account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH -p gengpu ## Required: (buyin, short, normal, long, gengpu, genhimem, etc)
#SBATCH --gres gpu:h100:4

#SBATCH -t 48:00:00 ## Required: How long will the job need to run (remember different partitions have restrictions on this parameter)
#SBATCH --nodes 1 ## how many computers/nodes do you need (no default)
#SBATCH --cpus-per-task 64
#SBATCH --mem 300G ## how much RAM do you need per computer/node (this affects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=debug_wan5b ## When you run squeue -u 
#SBATCH --mail-type END ## BEGIN, END, FAIL or ALL
#SBATCH --error=shang/debug_wan5b.err
#SBATCH --output=shang/debug_wan5b.out

module purge
module load moose/1.0.0
module load python-miniconda3/4.12.0
module load cuda/12.0.1-gcc-12.3.0 

source /software/miniconda3/4.12.0/etc/profile.d/conda.sh

conda activate /projects/p32294/conda_env/diffsyn

cd /projects/p32294/Sim4VideoGen
bash slurm_stage1.sh

