#!/usr/bin/env bash

echo "(run_singularity.py): Called on compute node from current isaaclab directory $1 with container profile $2 and arguments ${@:3}"

#==
# Helper functions
#==

setup_directories() {
    # Check and create directories with proper permissions
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            chmod 777 "$dir"  
            echo "Created directory with open permissions: $dir"
        fi
    done
    
    local texture_cache_dir="${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov/texturecache"
    mkdir -p "$texture_cache_dir"
    chmod 777 "$texture_cache_dir"
    echo "Ensured texture cache directory exists: $texture_cache_dir"
}


#==
# Main
#==

# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load environment variables
source "$SCRIPT_DIR/.env.cluster"
source "$SCRIPT_DIR/../.env.base"

# setup directories with proper permissions
setup_directories

# copy all cache files to tmp
echo "Copying cache files to $TMPDIR..."
cp -r "$CLUSTER_ISAAC_SIM_CACHE_DIR" "$TMPDIR"

# ensure logs directory exists
mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
touch "$CLUSTER_ISAACLAB_DIR/logs/.keep"

# copy the isaaclab directory to compute node
echo "Copying Isaac Lab directory to $TMPDIR..."
cp -r "$1" "$TMPDIR"
dir_name=$(basename "$1")

# extract container
echo "Extracting container $2..."
tar -xf "$CLUSTER_SIF_PATH/$2.tar" -C "$TMPDIR"

# create persistent overlay
echo "Creating overlay image..."
apptainer overlay create --size 4096 "$CLUSTER_ISAACLAB_DIR/$dir_name.img"

# Main execution with fixed paths and debug env vars
echo "Launching Singularity container..."
singularity exec \
<<<<<<< Updated upstream
    -B "$TMPDIR/docker-isaac-sim/cache/ov:/nethome/fwu91/.cache/ov:rw" \
    -B "$TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw" \
    -B "$TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw" \
    -B "$TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw" \
    -B "$TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw" \
    -B "$TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw" \
    -B "$TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw" \
    -B "$TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw" \
    -B "$TMPDIR/$dir_name:/workspace/isaaclab:rw" \
    -B "$CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/logs:rw" \
    --overlay "$CLUSTER_ISAACLAB_DIR/$dir_name.img" \
    --nv --containall "$TMPDIR/$2.sif" \
    bash -c "
    export OV_CACHE=/nethome/fwu91/.cache/ov
    export WANDB_API_KEY="2aad938943de58373db00ca95109cf2b510f0252" 
    export NVIDIA_VERBOSE=1
    export ISAACLAB_PATH=/workspace/isaaclab
    echo 'Container environment:'
    env | sort
    echo 'Texture cache directory contents:'
    ls -l /nethome/fwu91/.cache/ov/texturecache/
    cd /workspace/isaaclab && \
    /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} ${@:3}
    "
=======
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B $TMPDIR/$dir_name:/workspace/isaaclab:rw \
    -B $CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/logs:rw \
    --overlay $CLUSTER_ISAACLAB_DIR/$dir_name.img \
    --nv --containall $TMPDIR/$2.sif \
    bash -c "export WANDB_API_KEY=b2c00a2d706c11e8c3cca85ec3df3130f8dbf839 && \
            export ISAACLAB_PATH=/workspace/isaaclab && \
            cd /workspace/isaaclab && \
            /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} ${@:3}"
>>>>>>> Stashed changes

# sync back cache files
echo "Syncing cache files back to host..."
rsync -azPv "$TMPDIR/docker-isaac-sim" "$CLUSTER_ISAAC_SIM_CACHE_DIR/.."

# cleanup
if $REMOVE_CODE_COPY_AFTER_JOB; then
    echo "Removing temporary code copy..."
    rm -rf "$1"
fi

if $REMOVE_OVERLAY_AFTER_JOB; then
    echo "Removing overlay image..."
    rm -f "$CLUSTER_ISAACLAB_DIR/$dir_name.img"
fi

echo "(run_singularity.py): Execution completed"