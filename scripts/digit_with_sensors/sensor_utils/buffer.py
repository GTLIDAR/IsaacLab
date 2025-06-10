import torch
import os
import time
from typing import List, Optional
from PIL import Image
import numpy as np

class RGB8FrameBuffer:
    """Buffer system for managing RGB8 frames from forward-tis-dfm27up camera.
    
    Supports multi-environment inputs (shape: [n_envs, H, W, 3]).
    """
    
    def __init__(
        self,
        max_cpu_frames: int = 5000,      # Maximum frames to keep in CPU memory
        auto_flush_interval: int = 10,  # Process frames every N frames
        output_dir: str = "scripts/digit_with_sensors/output/rgb",
        device: str = "cuda",
        n_envs: int = 1,                # Number of parallel environments
    ):
        """Initialize the RGB frame buffer.
        
        Args:
            max_cpu_frames: Maximum frames stored in CPU memory before processing
            auto_flush_interval: Process frames automatically after N accumulations
            output_dir: Directory to save processed frames (subdirectories per env)
            device: GPU device being used ('cuda' or 'cuda:0')
            n_envs: Number of parallel environments (for organizing output)
        """
        self.max_cpu_frames = max_cpu_frames
        self.auto_flush_interval = auto_flush_interval
        self.output_dir = output_dir
        self.device = device
        self.n_envs = n_envs
        
        # Initialize buffer: List[ (env_id, frame_tensor) ]
        self.frame_buffer: List[tuple[int, torch.Tensor]] = []
        self.frame_count = 0
        self.total_processed = 0
        
        # Create output directories per environment
        for env_id in range(n_envs):
            os.makedirs(os.path.join(output_dir, f"env_{env_id}"), exist_ok=True)
    
    def add_frame(self, frame_data: torch.Tensor, env_ids: Optional[List[int]] = None):
        """Add batch of RGB frames to the buffer.
        
        Args:
            frame_data: RGB tensor (shape: [n_envs, H, W, 3] or [H, W, 3])
            env_ids: Optional list of environment IDs (default: range(n_envs))
        """
        # Ensure data is on GPU
        if not frame_data.is_cuda:
            frame_data = frame_data.to(self.device)
        
        # Handle single frame input
        if frame_data.dim() == 3:
            frame_data = frame_data.unsqueeze(0)  # [H,W,3] -> [1,H,W,3]
        
        # Default env_ids if not provided
        if env_ids is None:
            env_ids = list(range(frame_data.shape[0]))
        elif isinstance(env_ids, int):
            env_ids = [env_ids]
        
        # Split batch and add to buffer
        for env_id, env_frame in zip(env_ids, frame_data):
            self.frame_buffer.append((env_id, env_frame.cpu()))
            self.frame_count += 1
        
        # Auto-flush logic
        if len(self.frame_buffer) >= self.max_cpu_frames:
            self.flush()
        elif self.frame_count % self.auto_flush_interval == 0:
            self.flush()
    
    def flush(self):
        """Process all frames in the buffer and clear it."""
        if not self.frame_buffer:
            return
            
        start_time = time.time()
        processed_count = len(self.frame_buffer)
        
        # Process all frames in the buffer
        
        self.total_processed += processed_count
        self.frame_buffer.clear()
        
        print(f"Processed {processed_count} frames in {time.time()-start_time:.2f}s")
    
    def save_frame(
        self, 
        frame_tensor: torch.Tensor, 
        env_id: int, 
        frame_idx: int
    ):
        """Save a single RGB frame to disk with environment ID.
        
        Args:
            frame_tensor: RGB tensor in CPU memory (shape: [H,W,3])
            env_id: Environment ID for subdirectory
            frame_idx: Global frame index for filename
        """
        # Convert to numpy and scale to 0-255
        np_frame = (frame_tensor.numpy() * 255).astype(np.uint8)
        
        # Save to env-specific subdirectory
        env_dir = os.path.join(self.output_dir, f"env_{env_id}")
        os.makedirs(env_dir, exist_ok=True)
        Image.fromarray(np_frame).save(
            os.path.join(env_dir, f"frame_{frame_idx:06d}.jpg")
        )
    
    def __del__(self):
        """Destructor ensures all frames are processed before deletion."""
        if self.frame_buffer:
            print("Flushing remaining frames...")
            self.flush()