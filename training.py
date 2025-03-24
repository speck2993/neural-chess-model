import os
import numpy as np
import torch
import random
from collections import OrderedDict
import time
from datetime import datetime
import gc
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import logging
import glob
from model import ChessResNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("chess_training")

class LossTracker:
    """Tracks and plots training and validation metrics throughout training with detailed loss components"""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.metrics = {
            # Total losses
            'train_loss': [],
            'valid_loss': [],
            
            # Main policy losses
            'train_policy_loss': [],
            'valid_policy_loss': [],
            
            # Main value losses
            'train_value_loss': [],
            'valid_value_loss': [],
            
            # Auxiliary policy losses
            'train_aux_policy_loss': [],
            'valid_aux_policy_loss': [],
            
            # Auxiliary value losses
            'train_aux_value_loss': [],
            'valid_aux_value_loss': [],
            
            # Other metrics
            'valid_accuracy': [],
            'learning_rate': [],
            'iteration_times': [],
            'iterations': []
        }
        
        # Ensure metrics directory exists
        self.metrics_dir = os.path.join(save_dir, 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Generate a timestamp to identify this training run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metrics_file = os.path.join(self.metrics_dir, f'metrics_{self.timestamp}.json')
        
        # Create plots directory
        self.plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def update(self, iteration, train_metrics, valid_metrics=None, learning_rate=None, iteration_time=None):
        """Update metrics with new values from an iteration"""
        self.metrics['iterations'].append(iteration)
        
        # Update training metrics
        self.metrics['train_loss'].append(train_metrics.get('loss', 0))
        self.metrics['train_policy_loss'].append(train_metrics.get('policy_loss', 0))
        self.metrics['train_value_loss'].append(train_metrics.get('value_loss', 0))
        self.metrics['train_aux_policy_loss'].append(train_metrics.get('aux_policy_loss', 0))
        self.metrics['train_aux_value_loss'].append(train_metrics.get('aux_value_loss', 0))
        
        # Update validation metrics if provided
        if valid_metrics:
            self.metrics['valid_loss'].append(valid_metrics.get('loss', 0))
            self.metrics['valid_policy_loss'].append(valid_metrics.get('policy_loss', 0))
            self.metrics['valid_value_loss'].append(valid_metrics.get('value_loss', 0))
            self.metrics['valid_aux_policy_loss'].append(valid_metrics.get('aux_policy_loss', 0))
            self.metrics['valid_aux_value_loss'].append(valid_metrics.get('aux_value_loss', 0))
            self.metrics['valid_accuracy'].append(valid_metrics.get('outcome_accuracy', 0))
        
        # Update other metrics
        if learning_rate is not None:
            self.metrics['learning_rate'].append(learning_rate)
        
        if iteration_time is not None:
            self.metrics['iteration_times'].append(iteration_time)
        
        # Save metrics after each update
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def generate_plots(self, iteration):
        """Generate and save plots for the current metrics"""
        try:
            # Ensure plot directory exists
            os.makedirs(self.plots_dir, exist_ok=True)
            
            # Set Matplotlib to use non-interactive backend
            plt.switch_backend('agg')
            
            # Apply a nice style
            try:
                plt.style.use('seaborn-v0_8-pastel')
            except:
                plt.style.use('seaborn')
            
            # Create two separate plots for better visualization
            
            # PLOT 1: Total and Main Losses
            plt.figure(figsize=(12, 10))
            
            # Total Loss
            plt.subplot(2, 2, 1)
            train_loss = self.metrics['train_loss']
            valid_loss = self.metrics['valid_loss']
            iterations = self.metrics['iterations'][:len(train_loss)]
            plt.plot(iterations, train_loss, 'b-', label='Training')
            if valid_loss:
                valid_iterations = self.metrics['iterations'][:len(valid_loss)]
                plt.plot(valid_iterations, valid_loss, 'r-', label='Validation')
            plt.title('Total Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Main Policy Loss
            plt.subplot(2, 2, 2)
            train_policy = self.metrics['train_policy_loss']
            valid_policy = self.metrics['valid_policy_loss']
            iterations = self.metrics['iterations'][:len(train_policy)]
            plt.plot(iterations, train_policy, 'b-', label='Training')
            if valid_policy:
                valid_iterations = self.metrics['iterations'][:len(valid_policy)]
                plt.plot(valid_iterations, valid_policy, 'r-', label='Validation')
            plt.title('Main Policy Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Main Value Loss
            plt.subplot(2, 2, 3)
            train_value = self.metrics['train_value_loss']
            valid_value = self.metrics['valid_value_loss']
            iterations = self.metrics['iterations'][:len(train_value)]
            plt.plot(iterations, train_value, 'b-', label='Training')
            if valid_value:
                valid_iterations = self.metrics['iterations'][:len(valid_value)]
                plt.plot(valid_iterations, valid_value, 'r-', label='Validation')
            plt.title('Main Value Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Learning Rate
            plt.subplot(2, 2, 4)
            if self.metrics['learning_rate']:
                data = self.metrics['learning_rate']
                iterations = self.metrics['iterations'][:len(data)]
                plt.plot(iterations, data, 'g-')
                plt.title('Learning Rate')
                plt.xlabel('Iteration')
                plt.ylabel('Learning Rate')
                plt.yscale('log')  # Use log scale for learning rate
            elif self.metrics['valid_accuracy']:
                data = self.metrics['valid_accuracy']
                iterations = self.metrics['iterations'][:len(data)]
                plt.plot(iterations, data, 'g-')
                plt.title('Validation Accuracy')
                plt.xlabel('Iteration')
                plt.ylabel('Accuracy')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add a title to the entire plot
            plt.suptitle(f'Main Training Metrics - Iteration {iteration}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save the combined plot
            plot_path = os.path.join(self.plots_dir, f'main_metrics_iter{iteration}_{self.timestamp}.png')
            plt.savefig(plot_path, dpi=120)
            plt.close()
            
            # PLOT 2: Auxiliary Losses
            plt.figure(figsize=(12, 10))
            
            # Auxiliary Policy Loss
            plt.subplot(2, 2, 1)
            train_aux_policy = self.metrics['train_aux_policy_loss']
            valid_aux_policy = self.metrics['valid_aux_policy_loss']
            iterations = self.metrics['iterations'][:len(train_aux_policy)]
            plt.plot(iterations, train_aux_policy, 'b-', label='Training')
            if valid_aux_policy:
                valid_iterations = self.metrics['iterations'][:len(valid_aux_policy)]
                plt.plot(valid_iterations, valid_aux_policy, 'r-', label='Validation')
            plt.title('Auxiliary Policy Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Auxiliary Value Loss
            plt.subplot(2, 2, 2)
            train_aux_value = self.metrics['train_aux_value_loss']
            valid_aux_value = self.metrics['valid_aux_value_loss']
            iterations = self.metrics['iterations'][:len(train_aux_value)]
            plt.plot(iterations, train_aux_value, 'b-', label='Training')
            if valid_aux_value:
                valid_iterations = self.metrics['iterations'][:len(valid_aux_value)]
                plt.plot(valid_iterations, valid_aux_value, 'r-', label='Validation')
            plt.title('Auxiliary Value Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Policy Loss Comparison (Main vs Auxiliary)
            plt.subplot(2, 2, 3)
            train_policy = self.metrics['train_policy_loss']
            train_aux_policy = self.metrics['train_aux_policy_loss']
            iterations = self.metrics['iterations'][:len(train_policy)]
            plt.plot(iterations, train_policy, 'b-', label='Main Policy')
            plt.plot(iterations, train_aux_policy, 'g-', label='Aux Policy')
            plt.title('Policy Loss Comparison')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Value Loss Comparison (Main vs Auxiliary)
            plt.subplot(2, 2, 4)
            train_value = self.metrics['train_value_loss']
            train_aux_value = self.metrics['train_aux_value_loss']
            iterations = self.metrics['iterations'][:len(train_value)]
            plt.plot(iterations, train_value, 'b-', label='Main Value')
            plt.plot(iterations, train_aux_value, 'g-', label='Aux Value')
            plt.title('Value Loss Comparison')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add a title to the entire plot
            plt.suptitle(f'Auxiliary & Comparison Metrics - Iteration {iteration}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save the auxiliary plot
            aux_plot_path = os.path.join(self.plots_dir, f'aux_metrics_iter{iteration}_{self.timestamp}.png')
            plt.savefig(aux_plot_path, dpi=120)
            plt.close()
            
            logger.info(f"Generated metrics plots for iteration {iteration}")
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            
class ChessDataLoader:
    """Memory-optimized data loader for chess position data with enhanced randomization and file caching"""
    def __init__(self, data_dir, batch_size=512, shuffle_files=True, shuffle_positions=True, 
                 monitor_memory=False, use_cache=True, cache_file=None, seed=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.shuffle_positions = shuffle_positions
        self.monitor_memory = monitor_memory
        self.use_cache = use_cache
        self.seed = seed
        
        # Set random seed if provided (for reproducible shuffling)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        # Get list of all data files
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        
        # Set default cache file name if not provided
        if cache_file is None:
            self.cache_file = os.path.join(data_dir, 'dataloader_cache.json')
        else:
            self.cache_file = cache_file
        
        # Current file and position tracking
        self.current_file_idx = 0
        self.current_position = 0
        
        # For position randomization
        self.current_positions = None
        self.current_file_data = None
        
        # File cache - use OrderedDict to track insertion order
        self.data_cache = OrderedDict()  # {filename: data}
        self.max_cache_size = 3
        
        # Statistics for monitoring
        self.total_samples = 0
        self.files_info = {}  # {filename: num_samples}
        
        # Try to load cached file info if cache is enabled
        cache_loaded = False
        if self.use_cache:
            cache_loaded = self._load_file_info_cache()
            
        # If cache wasn't loaded or is disabled, scan files
        if not cache_loaded:
            # Calculate total samples and store file info
            logger.info(f"Scanning {len(self.files)} files to count samples...")
            for file in self.files:
                file_path = os.path.join(data_dir, file)
                try:
                    with np.load(file_path) as data:
                        num_samples = len(data['bitboards'])
                        self.files_info[file] = num_samples
                        self.total_samples += num_samples
                except Exception as e:
                    logger.error(f"Error loading file {file}: {str(e)}")
            
            # Save the file info to cache if enabled
            if self.use_cache:
                self._save_file_info_cache()
        
        # Shuffle files after loading all info if needed
        if shuffle_files:
            random.shuffle(self.files)
            
        logger.info(f"Initialized loader with {len(self.files)} files, {self.total_samples} total samples")
        
        # For memory monitoring
        if monitor_memory:
            try:
                import psutil
                self.psutil = psutil
                self.process = psutil.Process(os.getpid())
                self.last_memory = self.get_memory_usage()
                logger.info(f"Initial memory usage: {self.last_memory:.2f} MB")
            except ImportError:
                logger.warning("psutil not installed. Memory monitoring disabled.")
                self.monitor_memory = False
                
    def _save_file_info_cache(self):
        """Save file info to cache file"""
        try:
            cache_data = {
                'files_info': self.files_info,
                'total_samples': self.total_samples,
                'data_dir': self.data_dir,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f)
            logger.info(f"Saved file info cache to {self.cache_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving cache file: {str(e)}")
            return False
    
    def _load_file_info_cache(self):
        """Load file info from cache file"""
        if not os.path.exists(self.cache_file):
            logger.info(f"Cache file {self.cache_file} not found. Will scan files.")
            return False
            
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                
            # Verify cache is for the same data directory
            if cache_data.get('data_dir') != self.data_dir:
                logger.warning(f"Cache is for different data directory. Will rescan files.")
                return False
            
            # Check if all files in self.files exist in the cache
            all_files_present = all(file in cache_data['files_info'] for file in self.files)
            
            if not all_files_present:
                logger.warning("New files found that aren't in cache. Will rescan files.")
                return False
                
            # Load cache data
            self.files_info = cache_data['files_info']
            self.total_samples = cache_data['total_samples']
            
            cache_time = cache_data.get('timestamp', 'unknown time')
            logger.info(f"Loaded file info from cache created at {cache_time}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading cache file: {str(e)}")
            return False

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        if not self.monitor_memory:
            return 0
        return self.process.memory_info().rss / (1024 * 1024)
    
    def log_memory_usage(self, operation):
        """Log memory usage change"""
        if not self.monitor_memory:
            return
        
        current_memory = self.get_memory_usage()
        diff = current_memory - self.last_memory
        logger.info(f"Memory after {operation}: {current_memory:.2f} MB (Change: {diff:.2f} MB)")
        self.last_memory = current_memory

    def _load_file(self, filename):
        """Load a file into the cache, evicting oldest file if necessary"""
        start_time = time.time()
        
        # If cache is full, remove oldest item
        if len(self.data_cache) >= self.max_cache_size:
            # Get the oldest file and its data
            oldest_file, oldest_data = self.data_cache.popitem(last=False)
            
            # Explicitly delete the data to help garbage collection
            if self.monitor_memory:
                logger.debug(f"Removing {oldest_file} from cache")
            
            del oldest_data
            gc.collect()  # Encourage garbage collection
            
            if self.monitor_memory:
                self.log_memory_usage(f"removing {oldest_file}")
        
        # Load file
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            # Use with statement to ensure file is closed properly
            with np.load(file_path) as npz_file:
                # Convert to regular dict to avoid file handle issues
                data = {
                    'bitboards': npz_file['bitboards'].copy(),
                    'policies': npz_file['policies'].copy(),
                    'results': npz_file['results'].copy() if 'results' in npz_file else np.zeros(len(npz_file['bitboards']))
                }
            
            # Store in cache
            self.data_cache[filename] = data
            
            load_time = time.time() - start_time
            logger.debug(f"Loaded file {filename} in {load_time:.2f}s, cache has {len(self.data_cache)} files")
            
            if self.monitor_memory:
                self.log_memory_usage(f"loading {filename}")
            
            return data
        except Exception as e:
            logger.error(f"Error loading file {filename}: {str(e)}")
            return None
    
    def _get_current_file_data(self):
        """Get the data for the current file and setup position randomization"""
        if self.current_file_idx >= len(self.files):
            return None
            
        current_file = self.files[self.current_file_idx]
        
        # Load file if not in cache
        if current_file not in self.data_cache:
            data = self._load_file(current_file)
            if data is None:
                # Skip this file if loading failed
                self.current_file_idx += 1
                self.current_position = 0
                self.current_positions = None
                return self._get_current_file_data()
        else:
            # Move this file to the end of the OrderedDict (mark as most recently used)
            data = self.data_cache.pop(current_file)
            self.data_cache[current_file] = data
        
        # Setup randomized position access if needed
        if self.shuffle_positions and (self.current_positions is None or self.current_file_data != data):
            num_positions = len(data['bitboards'])
            
            # Use seed for consistent randomization if provided
            if self.seed is not None:
                # Create a deterministic but unique seed for each file
                file_seed = int(hash(current_file) % 2**32)
                combined_seed = (self.seed + file_seed) % 2**32
                prev_state = np.random.get_state()
                np.random.seed(combined_seed)
                self.current_positions = np.random.permutation(num_positions)
                np.random.set_state(prev_state)  # Restore random state
            else:
                self.current_positions = np.random.permutation(num_positions)
                
            self.current_file_data = data
            self.current_position = 0
            
        return data
    
    def next_batch(self):
        """Get the next batch of data with optimized memory handling and position randomization"""
        # Initialize empty numpy arrays instead of lists for better memory efficiency
        max_samples = self.batch_size
        bitboards = np.zeros((max_samples, 17, 8, 8), dtype=np.float32)
        policies = np.zeros((max_samples, 73, 8, 8), dtype=np.float32)
        results = np.zeros((max_samples, 3), dtype=np.float32)
        
        # Track how many samples we've filled
        filled_samples = 0
        
        # Fill the batch
        while filled_samples < max_samples:
            # Get current file data
            data = self._get_current_file_data()
            
            # If we've processed all files, return what we have or None
            if data is None:
                if filled_samples == 0:
                    return None
                break
                
            # Get number of samples in current file
            samples_in_file = len(data['bitboards'])
            
            # Calculate how many samples to take from this file
            samples_to_take = min(
                max_samples - filled_samples,  # How many more we need
                samples_in_file - self.current_position  # How many are left in file
            )
            
            # Use randomized indices if enabled
            if self.shuffle_positions:
                end_pos = self.current_position + samples_to_take
                indices = self.current_positions[self.current_position:end_pos]
                
                # Copy data using randomized indices
                for i, idx in enumerate(indices):
                    bitboards[filled_samples + i] = data['bitboards'][idx]
                    policies[filled_samples + i] = data['policies'][idx]
                    
                    # Process results into one-hot format
                    result = data['results'][idx]
                    if result == 1:  # White wins
                        results[filled_samples + i] = [1, 0, 0]
                    elif result == 0:  # Draw
                        results[filled_samples + i] = [0, 1, 0]
                    else:  # Black wins
                        results[filled_samples + i] = [0, 0, 1]
            else:
                # Sequential access
                end_pos = self.current_position + samples_to_take
                
                # Copy data directly to pre-allocated arrays
                bitboards[filled_samples:filled_samples+samples_to_take] = data['bitboards'][self.current_position:end_pos]
                policies[filled_samples:filled_samples+samples_to_take] = data['policies'][self.current_position:end_pos]
                
                # Process results into one-hot format
                for i in range(samples_to_take):
                    idx = self.current_position + i
                    result = data['results'][idx]
                    if result == 1:  # White wins
                        results[filled_samples + i] = [1, 0, 0]
                    elif result == 0:  # Draw
                        results[filled_samples + i] = [0, 1, 0]
                    else:  # Black wins
                        results[filled_samples + i] = [0, 0, 1]
            
            # Update position and filled count
            self.current_position += samples_to_take
            filled_samples += samples_to_take
            
            # If we've reached the end of the file, move to next file
            if self.current_position >= samples_in_file:
                self.current_file_idx += 1
                self.current_position = 0
                self.current_positions = None
        
        # Trim arrays to actual size if we didn't fill the batch
        if filled_samples < max_samples:
            bitboards = bitboards[:filled_samples]
            policies = policies[:filled_samples]
            results = results[:filled_samples]
        
        # Convert to tensors
        batch_bitboards = torch.tensor(bitboards, dtype=torch.float32)
        batch_policies = torch.tensor(policies, dtype=torch.float32)
        batch_results = torch.tensor(results, dtype=torch.float32)
        
        # Explicitly clear numpy arrays to help garbage collection
        del bitboards
        del policies
        del results
        
        if self.monitor_memory:
            self.log_memory_usage("creating batch")
        
        return batch_bitboards, batch_policies, batch_results
    
    def reset(self):
        """Reset the loader to start from the beginning with thorough cleanup"""
        self.current_file_idx = 0
        self.current_position = 0
        self.current_positions = None
        self.current_file_data = None
        
        # Clear cache with careful memory management
        for key in list(self.data_cache.keys()):
            if self.monitor_memory:
                logger.debug(f"Removing {key} from cache during reset")
            del self.data_cache[key]
        
        self.data_cache.clear()
        gc.collect()  # Force garbage collection
        
        # Run additional garbage collection cycles for thorough cleanup
        for _ in range(3):
            gc.collect()
        
        # Shuffle files if needed
        if self.shuffle_files:
            # Use the seed for consistent shuffling if provided
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self.files)
            if self.seed is not None:
                # Reset the random state after shuffling
                random.seed()
            
        logger.info("Reset data loader")
        
        if self.monitor_memory:
            self.log_memory_usage("reset")
    
    def get_progress(self):
        """Calculate progress through the dataset"""
        # Calculate samples processed in completed files
        completed_samples = sum(self.files_info.get(f, 0) for f in self.files[:self.current_file_idx])
        
        # Add samples from current file
        if self.current_file_idx < len(self.files):
            completed_samples += self.current_position
            
        # Calculate percentage
        progress = (completed_samples / self.total_samples) * 100 if self.total_samples > 0 else 0
        return progress, completed_samples, self.total_samples
    
def reset_amp_state(model, device='cuda'):
    # Move model to CPU to clear CUDA caches and contexts
    model.cpu()
    
    # Thorough CUDA cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    # Reset ALL model components - parameters, buffers, everything
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # Pay special attention to convolutional layers
            if hasattr(module, 'weight'):
                module.weight.data = module.weight.data.float()
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = module.bias.data.float()
    
    # Also reset batchnorm statistics which can be problematic
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if hasattr(module, 'running_mean'):
                module.running_mean = module.running_mean.float()
            if hasattr(module, 'running_var'):
                module.running_var = module.running_var.float()
    
    # Move back to original device
    model.to(device)
    
    # Ensure we're in correct mode
    model.train()

def evaluate_model(model, data_loader, device, use_mixed_precision=True):
    """Evaluate the model on the given data loader with mixed precision support"""
    # Reset the data loader
    data_loader.reset()
    
    # Use the model's evaluate method
    metrics = model.evaluate(data_loader, device, use_mixed_precision)
    
    return metrics

def create_validation_loader(data_dir, num_validation_files=5, batch_size=512, use_cache=True, seed=42):
    """Create a separate data loader for validation data"""
    # Get all files
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    
    # Make sure we don't ask for more files than exist
    num_validation_files = min(num_validation_files, len(all_files))
    
    # Use a fixed seed for deterministic validation set
    prev_state = random.getstate()
    random.seed(seed)
    validation_files = random.sample(all_files, num_validation_files)
    random.setstate(prev_state)  # Restore random state
    
    # Create a validation directory if it doesn't exist
    validation_dir = os.path.join(data_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)
    
    # Create symlinks or copies to validation files
    for file in validation_files:
        src_path = os.path.join(data_dir, file)
        dst_path = os.path.join(validation_dir, file)
        
        # Create symlink if it doesn't exist
        if not os.path.exists(dst_path):
            try:
                if os.name == 'nt':  # Windows
                    # Windows might require admin privileges for symlinks, so copy instead
                    import shutil
                    shutil.copy2(src_path, dst_path)
                else:  # Unix-like
                    os.symlink(src_path, dst_path)
            except Exception as e:
                logger.error(f"Error creating symlink for validation file {file}: {str(e)}")
    
    # Create the cache file path specific to validation
    cache_file = os.path.join(validation_dir, 'validation_cache.json')
    
    # Create and return the validation loader with the specified batch size and fixed seed
    valid_loader = ChessDataLoader(
        validation_dir, 
        batch_size=batch_size, 
        shuffle_files=False, 
        shuffle_positions=True,
        use_cache=use_cache,
        cache_file=cache_file,
        seed=seed  # Use the same seed for consistent validation
    )
    
    return valid_loader, validation_files


def create_training_loader(data_dir, validation_files, batch_size=512, use_cache=True, seed=None):
    """Create a data loader for training data, excluding validation files"""
    # Create a training directory with symlinks to all non-validation files
    training_dir = os.path.join(data_dir, 'training')
    os.makedirs(training_dir, exist_ok=True)
    
    # Get all files and filter out validation files
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    training_files = [f for f in all_files if f not in validation_files]
    
    # Create symlinks or copies to training files
    for file in training_files:
        src_path = os.path.join(data_dir, file)
        dst_path = os.path.join(training_dir, file)
        
        # Create symlink if it doesn't exist
        if not os.path.exists(dst_path):
            try:
                if os.name == 'nt':  # Windows
                    # Windows might require admin privileges for symlinks, so copy instead
                    import shutil
                    shutil.copy2(src_path, dst_path)
                else:  # Unix-like
                    os.symlink(src_path, dst_path)
            except Exception as e:
                logger.error(f"Error creating symlink for training file {file}: {str(e)}")
    
    # Create the cache file path specific to training
    cache_file = os.path.join(training_dir, 'training_cache.json')
    
    # Create and return the training loader with enhanced randomization and specified batch size
    return ChessDataLoader(
        training_dir, 
        batch_size=batch_size, 
        shuffle_files=True, 
        shuffle_positions=True,
        use_cache=use_cache,
        cache_file=cache_file,
        seed=seed  # Use provided seed or None for random shuffling
    )


def train_iterations(data_dir, model, num_iterations=1000, valid_every=1000, 
                    save_every=5000, max_iterations=100000, batch_size=512, 
                    validation_batch_size=8192, save_dir='checkpoints', resume_iteration=0,
                    use_mixed_precision=True, warmup_iterations=500, use_cache=True,
                    seed=None):
    """
    Train model for a fixed number of iterations (batches) rather than complete epochs,
    with learning rate warmup and mixed precision training
    
    Args:
        data_dir: Directory containing training data
        model: The ChessResNet model to train
        num_iterations: Number of iterations to train for in each validation cycle
        valid_every: Run validation every N iterations
        save_every: Save checkpoint every N iterations
        max_iterations: Maximum total iterations before stopping
        batch_size: Batch size for training
        validation_batch_size: Batch size for validation (can be larger than training batch size)
        save_dir: Directory to save checkpoints
        resume_iteration: Starting iteration number if resuming training
        use_mixed_precision: Whether to use mixed precision training
        warmup_iterations: Number of iterations for learning rate warmup
        use_cache: Whether to use caching for data loader file info
        seed: Random seed for reproducible data shuffling (None for random)
    
    Returns:
        Dictionary with training metrics
    """
    # Ensure checkpoint directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up file logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'training_log_{timestamp}.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Initialize metric tracker
    tracker = LossTracker(save_dir)
    
    # Initialize validation loader first, then training loader to avoid overlap
    logger.info("Setting up validation dataset...")
    valid_loader, validation_files = create_validation_loader(
        data_dir, 
        num_validation_files=4, 
        batch_size=validation_batch_size,
        use_cache=use_cache,
        seed=42  # Always use same seed for validation
    )
    
    logger.info("Setting up training dataset...")
    train_loader = create_training_loader(
        data_dir, 
        validation_files, 
        batch_size=batch_size,
        use_cache=use_cache,
        seed=seed
    )
    
    logger.info(f"Starting training with {train_loader.total_samples} samples")
    logger.info(f"Validation set: {valid_loader.total_samples} samples")
    logger.info(f"Training batch size: {batch_size}, Validation batch size: {validation_batch_size}")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Training on {device}")
    
    # Enable mixed precision if available
    if use_mixed_precision and torch.cuda.is_available():
        logger.info("Using mixed precision training")
    else:
        use_mixed_precision = False
        logger.info("Mixed precision not available, using full precision")
    
    # Initialize scheduler with optimized parameters
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, mode='min', factor=0.7, patience=2, verbose=True, min_lr=1e-6
    )
    
    # Get initial learning rate
    initial_lr = model.optimizer.param_groups[0]['lr']
    target_lr = initial_lr  # Save for warmup calculations
    
    # Record best validation score
    best_valid_loss = float('inf')
    
    # Current iteration counter
    current_iteration = resume_iteration
    
    # Log model info and hyperparameters
    model_info = {
        'filters': getattr(model, 'filters', 256),
        'blocks': getattr(model, 'blocks', 12),
        'batch_size': batch_size,
        'validation_batch_size': validation_batch_size,
        'initial_learning_rate': initial_lr,
        'weight_decay': 0.0005,
        'policy_weight': model.policy_weight,
        'value_weight': model.value_weight,
        'auxiliary_weight': model.auxiliary_weight,
        'gradient_clip': model.gradient_clip,
        'grad_accumulation_steps': model.grad_accumulation_steps,
        'mixed_precision': use_mixed_precision,
        'warmup_iterations': warmup_iterations,
        'use_cache': use_cache,
        'data_seed': seed
    }
    logger.info(f"Model configuration: {json.dumps(model_info, indent=2)}")
    
    # Save model info to file
    model_info_file = os.path.join(save_dir, f'model_info_{timestamp}.json')
    with open(model_info_file, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Main training loop
    while current_iteration < max_iterations:
        model.train()
        
        # Training phase
        start_time = time.time()
        logger.info(f"\n=== Starting iterations {current_iteration+1} to {min(current_iteration+num_iterations, max_iterations)} ===")
        
        # Reset metrics for this batch of iterations
        batch_count = 0
        total_samples = 0
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_aux_policy_loss = 0
        total_aux_value_loss = 0
        
        # Process batches in this training cycle
        iters_this_cycle = min(num_iterations, max_iterations - current_iteration)
        
        for _ in range(iters_this_cycle):
            # Learning rate warmup
            if current_iteration < warmup_iterations:
                # Linear warmup from 0.001 to target_lr
                warmup_factor = current_iteration / warmup_iterations
                warmup_lr = 0.001 + (target_lr - 0.001) * warmup_factor
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # Get next batch
            batch = train_loader.next_batch()
                
            # If we've reached the end of the dataset, reset and continue
            if batch is None:
                logger.info("Reached end of dataset, resetting data loader")
                train_loader.reset()
                batch = train_loader.next_batch()
                
                # If still None after reset, something is wrong
                if batch is None:
                    logger.error("Data loader returned None after reset")
                    break
            
            # Process batch with mixed precision and gradient accumulation
            metrics = model.training_step(batch, device, use_mixed_precision)
            
            # Update metrics
            batch_count += 1
            current_iteration += 1
            batch_size = batch[0].size(0)
            total_samples += batch_size
            total_loss += metrics['loss'] * batch_size
            total_policy_loss += metrics['policy_loss'] * batch_size
            total_value_loss += metrics['value_loss'] * batch_size
            total_aux_policy_loss += metrics['aux_policy_loss'] * batch_size
            total_aux_value_loss += metrics['aux_value_loss'] * batch_size
            
            # Log progress periodically
            if batch_count % 10 == 0:
                elapsed = time.time() - start_time
                iter_per_sec = batch_count / elapsed if elapsed > 0 else 0
                current_lr = model.optimizer.param_groups[0]['lr']
                
                logger.info(f"Iter {current_iteration} | "
                           f"LR: {current_lr:.6f} | "
                           f"Loss: {metrics['loss']:.4f} | "
                           f"{iter_per_sec:.2f} iter/s")
            
            # Check if we should break early for validation
            if current_iteration % valid_every == 0:
                break
        
        # Calculate average metrics for this cycle
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_policy_loss = total_policy_loss / total_samples
            avg_value_loss = total_value_loss / total_samples
            avg_aux_policy_loss = total_aux_policy_loss / total_samples
            avg_aux_value_loss = total_aux_value_loss / total_samples
        else:
            avg_loss = avg_policy_loss = avg_value_loss = 0
            
        # Store training metrics
        train_metrics = {
            'loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'samples': total_samples,
            'aux_policy_loss': avg_aux_policy_loss,
            'aux_value_loss': avg_aux_value_loss
        }
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        logger.info(f"Completed {batch_count} iterations in {elapsed_time:.2f}s | "
                   f"Avg Loss: {avg_loss:.4f} | "
                   f"Policy Loss: {avg_policy_loss:.4f} | "
                   f"Value Loss: {avg_value_loss:.4f}")
        
        # Run validation if needed
        if current_iteration % valid_every == 0:
            valid_start = time.time()
            logger.info("Running validation...")
            valid_metrics = evaluate_model(model, valid_loader, device, use_mixed_precision)
            valid_time = time.time() - valid_start
            # Reset AMP state completely - HOLY SHIT FUCK THIS
            reset_amp_state(model, device)
            
            logger.info(f"Validation: Loss: {valid_metrics['loss']:.4f} | "
                       f"Policy Loss: {valid_metrics['policy_loss']:.4f} | "
                       f"Value Loss: {valid_metrics['value_loss']:.4f} | "
                       f"Outcome Acc: {valid_metrics['outcome_accuracy']:.2f} | "
                       f"Time: {valid_time:.2f}s")
            
            # Update learning rate based on validation loss
            scheduler.step(valid_metrics['loss'])
            
            # Save best model
            if valid_metrics['loss'] < best_valid_loss:
                best_valid_loss = valid_metrics['loss']
                best_model_path = os.path.join(save_dir, f"model_best_{timestamp}.pt")
                try:
                    model.save(best_model_path)
                    logger.info(f"Saved new best model with validation loss: {best_valid_loss:.4f}")
                except Exception as e:
                    logger.error(f"Error saving best model: {str(e)}")
            
            # Update metrics tracker with validation
            current_lr = model.optimizer.param_groups[0]['lr']
            tracker.update(current_iteration, train_metrics, valid_metrics, current_lr, elapsed_time)
            
            # Generate plots
            tracker.generate_plots(current_iteration)
        else:
            # Update metrics tracker without validation
            current_lr = model.optimizer.param_groups[0]['lr']
            tracker.update(current_iteration, train_metrics, None, current_lr, elapsed_time)
        
        # Save regular checkpoint if needed
        if current_iteration % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f"model_iter{current_iteration}_{timestamp}.pt")
            try:
                model.save(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Error saving checkpoint: {str(e)}")
        
        # Force thorough garbage collection between cycles
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    logger.info(f"Training completed! Best validation loss: {best_valid_loss:.4f}")
    return {
        'final_iteration': current_iteration,
        'best_valid_loss': best_valid_loss
    }

def get_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint file in the given directory."""
    import os
    import glob
    import re
    
    # Pattern to match checkpoint files (model_iter* or model_best*)
    pattern = os.path.join(checkpoint_dir, "model_*.pt")
    
    # Get all matching files
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    
    # Return the most recent checkpoint file
    return checkpoint_files[0]

def main():
    """Main function to run training"""
    # Set up command line argument parsing
    import argparse
    parser = argparse.ArgumentParser(description='Train chess model with optimized parameters')
    parser.add_argument('--data_dir', type=str, default='data/processed-pgns', help='Directory containing training data')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--iterations', type=int, default=200000000, help='Total iterations to train for')
    parser.add_argument('--valid_every', type=int, default=5000, help='Validate every N iterations')
    parser.add_argument('--save_every', type=int, default=10000, help='Save checkpoint every N iterations')
    parser.add_argument('--filters', type=int, default=256, help='Number of filters in model')
    parser.add_argument('--blocks', type=int, default=12, help='Number of blocks in model')
    parser.add_argument('--initial_lr', type=float, default=0.004, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--validation_batch_size', type=int, default=8192, help='Batch size for validation')
    parser.add_argument('--warmup', type=int, default=800, help='Warmup iterations')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--no_mixed_precision', action='store_true', help='Disable mixed precision training')
    parser.add_argument('--resume_iteration', type=int, default=0, help='Iteration to resume from')
    parser.add_argument('--no_cache', action='store_true', help='Disable data loader caching')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible data shuffling')
    parser.add_argument('--auto_resume', action='store_true', help='Automatically resume from the latest checkpoint')
    
    args = parser.parse_args()
    
    # Create model with optimized hyperparameters
    model = ChessResNet(filters=args.filters, blocks=args.blocks, initial_lr=args.initial_lr)
    
    checkpoint_path = args.checkpoint
    resume_iteration = args.resume_iteration
    
    if args.auto_resume or (args.checkpoint is None and args.resume_iteration > 0):
        latest_checkpoint = get_latest_checkpoint(args.save_dir)
        if latest_checkpoint:
            checkpoint_path = latest_checkpoint
            
            # Try to extract iteration number from the filename
            import re
            iter_match = re.search(r'model_iter(\d+)_', os.path.basename(latest_checkpoint))
            if iter_match and resume_iteration == 0:
                resume_iteration = int(iter_match.group(1))
                logger.info(f"Auto-resuming from iteration {resume_iteration}")
            elif 'best' in latest_checkpoint and resume_iteration == 0:
                # Get iteration from validation metrics if available
                try:
                    metrics_files = glob.glob(os.path.join(args.save_dir, 'metrics', '*.json'))
                    if metrics_files:
                        import json
                        with open(sorted(metrics_files, key=os.path.getmtime)[-1], 'r') as f:
                            metrics_data = json.load(f)
                            if metrics_data['iterations']:
                                resume_iteration = metrics_data['iterations'][-1]
                                logger.info(f"Auto-resuming from best model at iteration {resume_iteration}")
                except Exception as e:
                    logger.warning(f"Could not determine iteration from metrics: {e}")
    
    if checkpoint_path:
        try:
            model.load(checkpoint_path)
            logger.info(f"Resumed training from checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return
    
    # Train model
    train_iterations(
        data_dir=args.data_dir,
        model=model,
        num_iterations=args.valid_every,  # Train this many iterations before validating
        valid_every=args.valid_every,
        save_every=args.save_every,
        max_iterations=args.iterations,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        save_dir=args.save_dir,
        resume_iteration=args.resume_iteration,
        use_mixed_precision=not args.no_mixed_precision,
        warmup_iterations=args.warmup,
        use_cache=not args.no_cache,
        seed=args.seed
    )

if __name__ == "__main__":
    main()