#!/usr/bin/env bash
cat <<EOT > job.sh
#!/bin/bash

#SBATCH --gpus-per-node=rtx_pro_6000_blackwell:1
#SBATCH -N1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fwu91@gatech.edu
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT
sbatch < job.sh
rm job.sh