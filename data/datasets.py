"""
Dataset for DINOv2 training with efficient sharding and corruption handling.
"""

import os
import torch
from torch.utils.data import IterableDataset
from PIL import Image, PngImagePlugin, ImageFile
import zipfile
import io
import random
import numpy as np
import json
import glob
import time
import pickle
import fcntl
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Iterator, Set
import torch.distributed as dist

from .transforms import TMEDinoTransforms


class MemoryEfficientShardedPathologyDataset(IterableDataset):
    """
    Memory-efficient dataset with proper sharding for distributed training.
    Handles corrupted files and supports resuming from checkpoints.
    
    Args:
        base_dir: Root directory containing zip files
        index_file: Path to dataset index pickle file
        worker_id: Worker ID (set dynamically)
        num_workers: Number of workers (set dynamically)
        rank: Distributed training rank
        world_size: Total number of processes
        seed: Random seed
        global_size: Size of global crops
        local_size: Size of local crops
        local_crop_scale: Scale range for local crops
        global_crop_scale: Scale range for global crops
        n_local_crops: Number of local crops per image
        mean: Normalization mean
        std: Normalization std
        corruptions_dir: Directory containing corruption logs
    """
    def __init__(
        self,
        base_dir: str,
        index_file: str = "dataset_index.pkl",
        worker_id: int = 0,
        num_workers: int = 1,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        global_size: int = 224,
        local_size: int = 96,
        local_crop_scale: tuple = (0.05, 0.4),
        global_crop_scale: tuple = (0.4, 1.0),
        n_local_crops: int = 2,
        mean: tuple = (0.6816, 0.5640, 0.7232),
        std: tuple = (0.1617, 0.1714, 0.1389),
        corruptions_dir: str = "corruption_results"
    ):
        super().__init__()
        self.base_dir = base_dir
        self.index_file = index_file
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.corruptions_dir = corruptions_dir
        
        # Set parameters for transforms
        self.global_size = global_size
        self.local_size = local_size
        self.n_local_crops = n_local_crops
        self.local_crop_scale = local_crop_scale
        self.global_crop_scale = global_crop_scale
        self.mean = mean
        self.std = std
        
        # Initialize transforms
        self.transforms = TMEDinoTransforms(
            local_size=local_size,
            global_size=global_size,
            local_crop_scale=local_crop_scale,
            global_crop_scale=global_crop_scale,
            n_local_crops=n_local_crops,
            mean=mean,
            std=std,
        )
        
        # Setup corruption logging
        self.corruption_log_file = "runtime_corrupted_files.json"
        self.corruption_lock_file = f"{self.corruption_log_file}.lock"
        if not os.path.exists(self.corruption_lock_file):
            with open(self.corruption_lock_file, 'w') as f:
                pass
        
        # Load pre-known corrupted files
        self.corrupted_zip_files = self._load_known_corrupted_files()
        print(f"Loaded {len(self.corrupted_zip_files)} known corrupted zip files to exclude")
        
        # Load metadata
        self.index_metadata = self._load_index_metadata()
        self.total_images = self.index_metadata['total_images']
        
        # Filter corrupted files
        self._filter_corrupted_zip_files()
        
        # Defer shard calculation
        self.shard_calculated = False
        self.worker_files = []
        self.worker_image_ranges = []
        
        # Resume capability
        self.samples_to_skip = 0
        
        # Rate limiting
        self.error_count = 0
        self.last_error_time = time.time()
        self.max_errors_per_minute = 10
    
    def set_worker_info(self, worker_id: int, num_workers: int):
        """Set actual worker ID and num_workers from PyTorch DataLoader."""
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.shard_calculated = False
        print(f"Worker info set: worker_id={worker_id}, num_workers={num_workers}")
    
    def set_resume_position(self, global_samples_processed: int):
        """
        Set how many samples to skip for resuming from checkpoint.
        
        Args:
            global_samples_processed: Total samples processed across all GPUs and workers
        """
        total_workers = self.world_size * self.num_workers
        self.samples_to_skip = global_samples_processed // total_workers
        print(f"Resume: Worker will skip {self.samples_to_skip} samples from its shard")
    
    def _load_known_corrupted_files(self) -> Set[str]:
        """Load pre-scanned corrupt file information from JSON files."""
        corrupted_zip_files = set()
        
        if not os.path.exists(self.corruptions_dir):
            print(f"Warning: Corruptions directory {self.corruptions_dir} does not exist")
            return corrupted_zip_files
        
        json_files = glob.glob(os.path.join(self.corruptions_dir, "*.json"))
        if not json_files:
            print(f"Warning: No corruption JSON files found in {self.corruptions_dir}")
            return corrupted_zip_files
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    corruptions = json.load(f)
                
                for item in corruptions:
                    if 'zip_path' in item and item.get('error_type') == 'BadZipFile':
                        corrupted_zip_files.add(item['zip_path'])
                        
                print(f"Processed {json_file}: found {len(corruptions)} corruptions")
            except Exception as e:
                print(f"Error loading corruption file {json_file}: {e}")
        
        return corrupted_zip_files
    
    def _filter_corrupted_zip_files(self):
        """Filter out known corrupted zip files from the dataset."""
        if not self.corrupted_zip_files:
            self.zip_files = self.index_metadata['zip_files']
            self.images_per_zip = self.index_metadata['images_per_zip']
            return
        
        filtered_zip_files = []
        filtered_images_per_zip = []
        filtered_count = 0
        filtered_zips = 0
        
        for i, zip_path in enumerate(self.index_metadata['zip_files']):
            if zip_path in self.corrupted_zip_files:
                filtered_count += self.index_metadata['images_per_zip'][i]
                filtered_zips += 1
                continue
            
            filtered_zip_files.append(zip_path)
            filtered_images_per_zip.append(self.index_metadata['images_per_zip'][i])
        
        self.zip_files = filtered_zip_files
        self.images_per_zip = filtered_images_per_zip
        
        if filtered_zips > 0:
            print(f"Filtered out {filtered_count} images from {filtered_zips} corrupted zip files")
    
    def _load_index_metadata(self):
        """Load only metadata about the index."""
        index_metadata_path = self.index_file.replace('.pkl', '_metadata.pkl')
        
        if not os.path.exists(index_metadata_path):
            print(f"Creating index metadata from {self.index_file}")
            self._create_index_metadata(self.index_file, index_metadata_path)
        
        with open(index_metadata_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def _create_index_metadata(index_path, metadata_path):
        """Create lightweight metadata file from full index."""
        with open(index_path, 'rb') as f:
            all_index = pickle.load(f)
        
        metadata = {
            'total_images': 0,
            'zip_files': [],
            'images_per_zip': []
        }
        
        for zip_path, image_names in all_index:
            num_images = len(image_names)
            if num_images > 0:
                metadata['zip_files'].append(zip_path)
                metadata['images_per_zip'].append(num_images)
                metadata['total_images'] += num_images
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def _calculate_worker_shard(self):
        """Calculate which files and image indices belong to this worker."""
        global_worker_id = self.rank * self.num_workers + self.worker_id
        total_workers = self.world_size * self.num_workers
        
        self.worker_files = []
        self.worker_image_ranges = []
        
        worker_indices = []
        
        for zip_idx, (zip_path, num_images) in enumerate(zip(self.zip_files, self.images_per_zip)):
            if zip_path in self.corrupted_zip_files:
                continue
                
            zip_seed = self.seed + hash(zip_path) % 10000
            rng = random.Random(zip_seed)
            
            for img_idx in range(num_images):
                worker = rng.randint(0, total_workers - 1)
                if worker == global_worker_id:
                    worker_indices.append((zip_idx, img_idx))
        
        if worker_indices:
            worker_indices.sort()
            
            current_zip = worker_indices[0][0]
            start_img = worker_indices[0][1]
            
            for i, (zip_idx, img_idx) in enumerate(worker_indices[1:] + [(None, None)]):
                if zip_idx != current_zip or zip_idx is None:
                    self.worker_files.append(self.zip_files[current_zip])
                    self.worker_image_ranges.append((start_img, worker_indices[i][1] + 1))
                    
                    if zip_idx is not None:
                        current_zip = zip_idx
                        start_img = img_idx
        
        self.shard_calculated = True
        print(f"Worker {global_worker_id}/{total_workers} will process {len(worker_indices)} images from {len(self.worker_files)} zip files")
    
    def _get_image_names(self, zip_path, start_idx, end_idx):
        """Get specific image names for a range within a zip file."""
        if zip_path in self.corrupted_zip_files:
            return []
            
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # all_images = [f for f in zf.namelist() 
                #             if f.endswith('.png') 
                #             and ('_448_' in f) 
                #             and ('_224_' not in f)]
                all_images = [f for f in zf.namelist() 
                            if f.endswith('.webp')
                            and not f.startswith('__MACOSX')]
                all_images.sort()
                
                image_names = all_images[start_idx:end_idx]
            
            return image_names
        except Exception as e:
            print(f"Error reading zip file {zip_path}: {e}")
            self._log_corrupt_file(zip_path, "", e)
            self.corrupted_zip_files.add(zip_path)
            return []
    
    def _log_corrupt_file(self, zip_path, image_name, exception):
        """Log a corrupt file to JSON log with proper locking."""
        current_time = time.time()
        if current_time - self.last_error_time >= 60:
            self.error_count = 0
            self.last_error_time = current_time
            
        if self.error_count >= self.max_errors_per_minute:
            return
            
        self.error_count += 1
        
        corrupt_entry = {
            'zip_path': zip_path,
            'image_name': image_name,
            'error_type': type(exception).__name__,
            'error_msg': str(exception),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(self.corruption_lock_file, 'r+') as lockf:
                try:
                    fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    try:
                        try:
                            with open(self.corruption_log_file, 'r') as f:
                                existing_data = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError):
                            existing_data = []
                        
                        existing_data.append(corrupt_entry)
                        
                        with open(self.corruption_log_file, 'w') as f:
                            json.dump(existing_data, f, indent=2)
                    finally:
                        fcntl.flock(lockf, fcntl.LOCK_UN)
                except BlockingIOError:
                    pass
        except Exception as e:
            print(f"Error logging corrupt file: {e}")
    
    def _load_image(self, zip_path, image_name):
        """Load a single image from a zip file with error handling."""
        if zip_path in self.corrupted_zip_files:
            raise IOError(f"Skipping image from known corrupted zip file: {zip_path}")
            
        PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 * 1024)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        max_retries = 3
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    data = zf.read(image_name)
                    try:
                        img = Image.open(io.BytesIO(data)).convert('RGB')
                        return img
                    except ValueError as e:
                        if "Decompressed data too large" in str(e):
                            buffer = io.BytesIO(data)
                            img = Image.open(buffer)
                            img.load()
                            return img.convert('RGB')
                        raise
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
        
        current_time = time.time()
        if current_time - self.last_error_time >= 60:
            self.error_count = 0
            self.last_error_time = current_time
            
        if self.error_count < self.max_errors_per_minute:
            print(f"Failed to load image {image_name} from {zip_path} after {max_retries} attempts: {str(last_exception)}")
            self.error_count += 1
        
        self._log_corrupt_file(zip_path, image_name, last_exception)
        
        if isinstance(last_exception, (zipfile.BadZipFile, zipfile.LargeZipFile)):
            self.corrupted_zip_files.add(zip_path)
        
        raise IOError(f"Failed to load image after {max_retries} attempts: {str(last_exception)}")

    def __iter__(self):
        """Iterator with memory-efficient shuffling and resume support."""
        if not self.shard_calculated:
            self._calculate_worker_shard()
        
        rng = random.Random(self.seed + self.worker_id)
        
        shuffled_files = list(enumerate(self.worker_files))
        rng.shuffle(shuffled_files)
        
        samples_yielded = 0
        
        for file_idx, zip_path in shuffled_files:
            if zip_path in self.corrupted_zip_files:
                continue
                
            start_idx, end_idx = self.worker_image_ranges[file_idx]
            
            try:
                image_names = self._get_image_names(zip_path, start_idx, end_idx)
                
                if not image_names:
                    continue
                    
                rng.shuffle(image_names)
                
                for img_name in image_names:
                    if samples_yielded < self.samples_to_skip:
                        samples_yielded += 1
                        continue
                    
                    try:
                        img = self._load_image(zip_path, img_name)
                        crops = self.transforms(img)
                        samples_yielded += 1
                        yield crops
                    except Exception as e:
                        if isinstance(e, IOError) and "BadZipFile" in str(e):
                            self.corrupted_zip_files.add(zip_path)
                            break
                            
                        if self.error_count < self.max_errors_per_minute:
                            print(f"Skipping corrupted image {img_name} from {zip_path}: {e}")
                            self.error_count += 1
                        continue
                        
            except Exception as e:
                if self.error_count < self.max_errors_per_minute:
                    print(f"Error processing zip file {zip_path}: {e}")
                    self.error_count += 1
                self.corrupted_zip_files.add(zip_path)
    
    def __len__(self):
        """Return number of samples this worker will process."""
        total = 0
        for i, (start_idx, end_idx) in enumerate(self.worker_image_ranges):
            if i < len(self.worker_files) and self.worker_files[i] in self.corrupted_zip_files:
                continue
            total += (end_idx - start_idx)
        return total


class DINOv2PathologyDataset(torch.utils.data.IterableDataset):
    """
    Optimized dataset for DINOv2 training that returns pre-augmented views.
    Wrapper around MemoryEfficientShardedPathologyDataset.
    
    Args:
        base_dir: Root directory containing zip files
        index_file: Path to dataset index
        n_standard_local_crops: Number of standard local crops
        global_views: Number of global views
        local_crop_size: Size of local crops
        worker_id: Worker ID
        num_workers: Number of workers
        rank: Distributed training rank
        world_size: Total processes
        seed: Random seed
        global_size: Size of global crops
        mean: Normalization mean
        std: Normalization std
    """
    def __init__(
        self,
        base_dir: str,
        index_file: str,
        n_standard_local_crops: int,
        global_views: int,
        local_crop_size: int = 96,
        worker_id: int = 0,
        num_workers: int = 1,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        global_size: int = 224,
        mean: tuple = (0.6816, 0.5640, 0.7232),
        std: tuple = (0.1617, 0.1714, 0.1389),
    ):
        self.n_standard_local_crops = n_standard_local_crops
        self.global_views = global_views
        self.local_crop_size = local_crop_size
        
        actual_global_views = max(2, global_views)
        
        self.base_dataset = MemoryEfficientShardedPathologyDataset(
            base_dir=base_dir,
            index_file=index_file,
            worker_id=worker_id,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            seed=seed,
            global_size=global_size,
            local_size=local_crop_size,
            n_local_crops=n_standard_local_crops,
            mean=mean,
            std=std,
        )
    
    def __iter__(self):
        """Return all pre-augmented views flexibly."""
        for crops in self.base_dataset:
            output = []
            
            for i in range(self.global_views):
                if i < len(crops):
                    output.append(crops[i])
            
            for i in range(self.n_standard_local_crops):
                idx = 2 + i
                if idx < len(crops) - 1:
                    output.append(crops[idx])
            
            output.append(crops[-1])
            
            yield tuple(output)

    def __len__(self):
        return len(self.base_dataset)

    def set_resume_position(self, global_samples_processed: int):
        """Set resume position for checkpoint recovery."""
        if hasattr(self, 'base_dataset'):
            self.base_dataset.set_resume_position(global_samples_processed)
            print(f"DINOv2PathologyDataset: Set resume position to {global_samples_processed} samples")



# ============================================================================
# Multi-Dataset Wrapper with Proportional Sampling
# ============================================================================

class ProportionalMultiDatasetWrapper(IterableDataset):
    """
    Combines multiple dataset sources with proportional sampling per batch.
    Each batch maintains the same proportion as overall dataset distribution.
    
    Example: If datasets are 30% TCGA, 10% CPTAC, 60% IMPACT, then each
    batch of size 32 will have ~10 TCGA, ~3 CPTAC, ~19 IMPACT samples.
    """
    def __init__(
        self,
        dataset_configs: List[Dict],
        batch_size_per_gpu: int,
        worker_id: int = 0,
        num_workers: int = 1,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        
        self.batch_size_per_gpu = batch_size_per_gpu
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        
        # Initialize individual datasets
        self.datasets = []
        self.dataset_names = []
        self.dataset_sizes = []
        
        print("\n" + "="*80)
        print("Initializing Multi-Dataset with Proportional Sampling")
        print("="*80)
        
        for config in dataset_configs:
            name = config['name']
            base_dir = config['base_dir']
            index_file = config['index_file']
            
            print(f"\nLoading {name} dataset from {base_dir}...")
            
            dataset = MemoryEfficientShardedPathologyDataset(
                base_dir=base_dir,
                index_file=index_file,
                worker_id=worker_id,
                num_workers=num_workers,
                rank=rank,
                world_size=world_size,
                seed=seed,
                **kwargs
            )
            
            self.datasets.append(dataset)
            self.dataset_names.append(name)
            self.dataset_sizes.append(dataset.index_metadata['total_images'])
            
            print(f"  {name}: {dataset.index_metadata['total_images']:,} images")
        
        # Calculate proportions
        total_images = sum(self.dataset_sizes)
        self.proportions = [size / total_images for size in self.dataset_sizes]
        
        print("\n" + "-"*80)
        print("Dataset Proportions:")
        for name, size, prop in zip(self.dataset_names, self.dataset_sizes, self.proportions):
            print(f"  {name}: {size:,} images ({prop*100:.2f}%)")
        print(f"Total: {total_images:,} images")
        
        # Calculate samples per dataset per batch
        self.samples_per_dataset = self._calculate_batch_distribution()
        
        print("\n" + "-"*80)
        print(f"Per-batch distribution (batch_size={batch_size_per_gpu}):")
        for name, count in zip(self.dataset_names, self.samples_per_dataset):
            print(f"  {name}: {count} samples per batch ({count/batch_size_per_gpu*100:.1f}%)")
        print("="*80 + "\n")
        
        # Create iterators
        self.iterators = None
    
    def _calculate_batch_distribution(self):
        """
        Calculate how many samples from each dataset per batch.
        Ensures proportions are maintained and sum equals batch_size.
        """
        # Calculate ideal samples (may be fractional)
        ideal_samples = [prop * self.batch_size_per_gpu for prop in self.proportions]
        
        # Round to integers (floor first)
        samples = [int(s) for s in ideal_samples]
        
        # Distribute remaining samples to maintain sum = batch_size
        remainder = self.batch_size_per_gpu - sum(samples)
        
        # Give remaining samples to datasets with largest fractional parts
        fractional_parts = [(ideal - actual, idx) 
                           for idx, (ideal, actual) in enumerate(zip(ideal_samples, samples))]
        fractional_parts.sort(reverse=True)
        
        for i in range(remainder):
            idx = fractional_parts[i][1]
            samples[idx] += 1
        
        assert sum(samples) == self.batch_size_per_gpu, \
            f"Batch distribution error: {sum(samples)} != {self.batch_size_per_gpu}"
        
        return samples
    
    def set_worker_info(self, worker_id, num_workers):
        """Propagate worker info to all datasets"""
        self.worker_id = worker_id
        self.num_workers = num_workers
        for dataset in self.datasets:
            dataset.set_worker_info(worker_id, num_workers)
    
    def set_resume_position(self, global_samples_processed):
        """Propagate resume position to all datasets"""
        for dataset in self.datasets:
            dataset.set_resume_position(global_samples_processed)
    
    def __iter__(self):
        """
        Yield samples in proportion-maintaining pattern.
        Pattern repeats every batch_size samples.
        """
        # Create fresh iterators
        self.iterators = [iter(ds) for ds in self.datasets]
        
        # Create sampling pattern for one batch
        # Example: [0, 0, 0, ...(10x), 1, 1, 1 (3x), 2, 2, ...(19x)]
        pattern = []
        for dataset_idx, count in enumerate(self.samples_per_dataset):
            pattern.extend([dataset_idx] * count)
        
        # Shuffle pattern to avoid systematic bias within batch
        rng = random.Random(self.seed + self.rank * self.num_workers + self.worker_id)
        
        # Yield samples according to pattern
        while True:
            # Shuffle pattern for this batch
            batch_pattern = pattern.copy()
            rng.shuffle(batch_pattern)
            
            for dataset_idx in batch_pattern:
                try:
                    sample = next(self.iterators[dataset_idx])
                    yield sample
                except StopIteration:
                    # One dataset exhausted - recreate its iterator
                    print(f"Dataset {self.dataset_names[dataset_idx]} exhausted, restarting...")
                    self.iterators[dataset_idx] = iter(self.datasets[dataset_idx])
                    sample = next(self.iterators[dataset_idx])
                    yield sample
    
    def __len__(self):
        """Return combined length"""
        return sum(len(ds) for ds in self.datasets)