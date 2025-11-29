#!/usr/bin/env bash
cat <<EOT > job.sh
#!/bin/bash

#SBATCH --partition="wu-lab"
#SBATCH --gpus-per-node="a40:1"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=48G
#SBATCH --cpus-per-task=15
#SBATCH --time=12:00:00
#SBATCH --qos="short"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fwu91@gatech.edu
#SBATCH --job-name=IsaacLab-training-$(date +"%Y-%m-%d-%H:%M")-%j
#SBATCH --output=/coc/flash12/fwu91/IsaacLab/IsaacLab-training-$(date +"%Y-%m-%d-%H:%M")-%j/output.log
#SBATCH --error=/coc/flash12/fwu91/IsaacLab/IsaacLab-training-$(date +"%Y-%m-%d-%H:%M")-%j/error.log

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT
cat < job.sh
sbatch < job.sh
