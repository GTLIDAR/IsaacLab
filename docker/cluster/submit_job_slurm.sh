#!/usr/bin/env bash
cat <<EOT > job.sh
#!/bin/bash

#SBATCH --partition=wu-lab
#SBATCH --account=fwu91
#SBATCH --gpus-per-node="a40:1"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=48G
#SBATCH --cpus-per-task=15
#SBATCH --time=12:00:00
#SBATCH --qos=short
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fwu91@gatech.edu
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --output="output.log"
#SBATCH --error="error.log"
#SBATCH --nodelist=synapse,dendrite

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT
cat < job.sh
sbatch < job.sh
rm job.sh
