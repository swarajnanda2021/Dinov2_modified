# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")


def load_pretrained_linear_weights(linear_classifier, model_name, patch_size):
    url = None
    if model_name == "vit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth"
    elif model_name == "vit_small" and patch_size == 8:
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth"
    elif model_name == "resnet50":
        url = "dino_resnet50_pretrain/dino_resnet50_linearweights.pth"
    if url is not None:
        print("We load the reference pretrained linear weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)["state_dict"]
        linear_classifier.load_state_dict(state_dict, strict=True)
    else:
        print("We use random linear weights.")


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer_epochwise(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None

def cancel_gradients_last_layer(current_iteration, model, freeze_for_iters):
    """
    Cancel gradients in the last layer for a specified number of iterations
    
    Args:
        current_iteration (int): Current training iteration
        model: The model whose last layer gradients need to be canceled
        freeze_for_iters (int): Number of iterations to freeze the last layer for
    """
    if current_iteration >= freeze_for_iters:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint_epochwise(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    
    Args:
        ckp_path (str): Path to the checkpoint file
        run_variables (dict): Dictionary of variables to restore (e.g., iteration count)
        **kwargs: Dictionary of objects to restore with their corresponding checkpoint keys
    """
    if not os.path.isfile(ckp_path):
        return
    print(f"Found checkpoint at {ckp_path}")

    # Load checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)

    # Handle backwards compatibility with epoch-based checkpoints
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]
            elif var_name == "iteration" and "epoch" in checkpoint:
                # Convert old epoch-based checkpoint to iteration-based
                # Assuming standard dataset length and batch size
                run_variables[var_name] = checkpoint["epoch"] * len(kwargs.get("train_loader", []))
                print(f"Converting epoch {checkpoint['epoch']} to iteration {run_variables[var_name]}")
        
        # Add last_mask_update_iteration handling for continuous training
        if 'iteration' in run_variables and 'last_mask_update_iteration' in checkpoint:
            # If last_mask_update_iteration is in the checkpoint, add it to run_variables
            if 'last_mask_update_iteration' not in run_variables:
                run_variables['last_mask_update_iteration'] = checkpoint['last_mask_update_iteration']
        elif 'iteration' in run_variables and 'args' in checkpoint:
            # If not present but we have iteration and args, calculate it based on mask_update_freq
            if hasattr(checkpoint['args'], 'mask_update_freq'):
                current_iter = run_variables['iteration']
                mask_freq = checkpoint['args'].mask_update_freq
                if 'last_mask_update_iteration' not in run_variables:
                    run_variables['last_mask_update_iteration'] = current_iter - (current_iter % mask_freq)
                    print(f"Calculated last_mask_update_iteration as {run_variables['last_mask_update_iteration']}")

    # Load state dictionaries for models and optimizers
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(f"=> loaded '{key}' from checkpoint '{ckp_path}' with msg {msg}")
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print(f"=> loaded '{key}' from checkpoint: '{ckp_path}'")
                except ValueError:
                    print(f"=> failed to load '{key}' from checkpoint: '{ckp_path}'")
        else:
            print(f"=> key '{key}' not found in checkpoint: '{ckp_path}'")



def restart_from_checkpoint_temp(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    
    Args:
        ckp_path (str): Path to the checkpoint file
        run_variables (dict): Dictionary of variables to restore (e.g., iteration count)
        **kwargs: Dictionary of objects to restore with their corresponding checkpoint keys
    """
    if not os.path.isfile(ckp_path):
        return
    print(f"Found checkpoint at {ckp_path}")

    # Load checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)

    # Handle backwards compatibility with epoch-based checkpoints
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]
            elif var_name == "iteration" and "epoch" in checkpoint:
                # Convert old epoch-based checkpoint to iteration-based
                # Assuming standard dataset length and batch size
                run_variables[var_name] = checkpoint["epoch"] * len(kwargs.get("train_loader", []))
                print(f"Converting epoch {checkpoint['epoch']} to iteration {run_variables[var_name]}")

    # Load state dictionaries for models and optimizers
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(f"=> loaded '{key}' from checkpoint '{ckp_path}' with msg {msg}")
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print(f"=> loaded '{key}' from checkpoint: '{ckp_path}'")
                except ValueError:
                    print(f"=> failed to load '{key}' from checkpoint: '{ckp_path}'")
        else:
            print(f"=> key '{key}' not found in checkpoint: '{ckp_path}'")


def cosine_scheduler_epochwise(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def cosine_scheduler(base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
    """
    Creates a cosine learning rate schedule with warmup.
    
    Args:
        base_value (float): Initial value after warmup (or initial value if no warmup)
        final_value (float): Final value at the end of training
        total_iters (int): Total number of training iterations
        warmup_iters (int): Number of warmup iterations
        start_warmup_value (float): Starting value for warmup phase
        
    Returns:
        np.array: Schedule of values for each iteration
    """
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    remaining_iters = total_iters - warmup_iters
    if remaining_iters <= 0:
        return warmup_schedule

    iters = np.arange(remaining_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_iters
    return schedule

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

import time
import datetime
import torch
from collections import defaultdict, deque

class SmoothedValue(object):
    """Tracks a series of values and provides smoothed values over a window."""
    def __init__(self, window_size=20, fmt='{avg:.6f}'):
        self.window_size = window_size
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item() if len(d) > 0 else 0

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0

    def __str__(self):
        return self.fmt.format(avg=self.avg)

class IterationMetricLogger(object):
    """Logs metrics using iterations as the time unit, including GPU memory usage."""
    def __init__(self, total_iterations, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.total_iterations = total_iterations
        self.start_time = time.time()
        self.last_time = self.start_time

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        meter_strs = []
        for name, meter in self.meters.items():
            # Print current average and global average.
            meter_strs.append(f"{name}: {meter} ({meter.global_avg:.6f})")
        return self.delimiter.join(meter_strs)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            if hasattr(meter, 'synchronize_between_processes'):
                meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Yields elements from 'iterable' and logs metrics every 'print_freq' iterations,
        including ETA and GPU memory usage (if available).
        """
        i = 0
        if header is None:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f} MB'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    memory = torch.cuda.max_memory_allocated() / MB
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=memory))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s/it)'.format(
            header, total_time_str, total_time / len(iterable)))



def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def create_hybrid_process_groups(args):
    """
    Create process groups for hybrid pipeline + data parallelism.
    
    Topology:
    - Pipeline group: GPUs within same node (ranks that share model shards)
    - Data group: Corresponding GPUs across nodes (data parallel replicas)
    
    Example with 2 nodes x 4 GPUs = 8 total GPUs:
    Global ranks:  [0, 1, 2, 3,  4, 5, 6, 7]
                   └─ node 0 ─┘ └─ node 1 ─┘
    
    Pipeline groups: [0,1,2,3], [4,5,6,7]  (within-node)
    Data groups: [0,4], [1,5], [2,6], [3,7]  (across-node, same local rank)
    
    Args:
        args: Training arguments with world_size, rank, gpus_per_node, num_nodes
    
    Returns:
        pipeline_group: Process group for pipeline communication
        data_group: Process group for data parallel gradient synchronization
        local_rank: GPU rank within node (0 to gpus_per_node-1)
        node_id: Which node this rank belongs to
    """

    if not args.use_pipeline_parallel:
        # No pipeline parallelism, return None groups
        return None, None, args.gpu, args.rank // args.gpus_per_node
    
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    # Calculate node and local rank
    node_id = global_rank // args.gpus_per_node
    local_rank = global_rank % args.gpus_per_node

    # Verify topology
    expected_world_size = args.num_nodes * args.gpus_per_node
    if world_size != expected_world_size:
        raise ValueError(
            f"World size mismatch: got {world_size}, "
            f"expected {args.num_nodes} nodes x {args.gpus_per_node} GPUs = {expected_world_size}"
        )
    
    # ========== Create pipeline groups (within each node) ==========
    pipeline_groups = []
    for node in range(args.num_nodes):
        ranks = [node * args.gpus_per_node + i for i in range(args.gpus_per_node)]
        group = dist.new_group(ranks=ranks)
        pipeline_groups.append(group)
        
        if global_rank in ranks:
            pipeline_group = group
            if is_main_process() or global_rank == ranks[0]:
                print(f"Pipeline group for node {node}: ranks {ranks}")
    
    # ========== Create data parallel groups (across nodes) ==========
    data_groups = []
    for local_r in range(args.gpus_per_node):
        ranks = [node * args.gpus_per_node + local_r for node in range(args.num_nodes)]
        group = dist.new_group(ranks=ranks)
        data_groups.append(group)
        
        if global_rank in ranks:
            data_group = group
            if local_r == 0 or is_main_process():
                print(f"Data parallel group for local_rank {local_r}: ranks {ranks}")
    
    print(f"Rank {global_rank}: node={node_id}, local_rank={local_rank}, "
          f"pipeline_group_size={dist.get_world_size(pipeline_group)}, "
          f"data_group_size={dist.get_world_size(data_group)}")
    
    return pipeline_group, data_group, local_rank, node_id

def send_tensor(tensor, dst_rank, tag=0):
    """Send tensor to destination rank."""
    dist.send(tensor.contiguous(), dst=dst_rank, tag=tag)


def recv_tensor(tensor_shape, dtype, device, src_rank, tag=0):
    """Receive tensor from source rank."""
    tensor = torch.empty(tensor_shape, dtype=dtype, device=device)
    dist.recv(tensor, src=src_rank, tag=tag)
    return tensor

"""
Pipeline parallelism checkpoint utilities.
Handles saving/loading checkpoints across distributed pipeline stages.
"""

import os
import torch
import torch.distributed as dist
import random
import numpy as np


def save_pipeline_checkpoint(
    student_stage,
    teacher_stage,
    prototype_bank,
    optimizer_student,
    optimizer_prototypes,
    fp16_scaler,
    loss_modules,
    iteration,
    dataset_position,
    args,
    output_dir,
    local_rank,
    pipeline_group,
    data_group,
):
    """
    Save pipeline checkpoint with proper distributed coordination.
    
    Strategy:
    1. Each rank saves its own stage state to CPU
    2. Gather all stage states to rank 0 of each data parallel group
    3. Rank 0 saves complete checkpoint
    
    Args:
        student_stage: Student pipeline stage (DDP wrapped)
        teacher_stage: Teacher pipeline stage (DDP wrapped)
        prototype_bank: Prototype bank (only on last stage, can be None)
        optimizer_student: Student optimizer
        optimizer_prototypes: Prototype optimizer
        fp16_scaler: Mixed precision scaler
        loss_modules: Dict of loss modules
        iteration: Current iteration
        dataset_position: Current dataset position
        args: Training arguments
        output_dir: Output directory
        local_rank: Local GPU rank within node
        pipeline_group: Pipeline process group
        data_group: Data parallel process group
    """
    
    # Only save from rank 0 of each pipeline group (i.e., first GPU of each node)
    # This ensures we save one checkpoint per data parallel replica
    is_saver = (local_rank == 0)
    
    if not is_saver:
        # Other ranks in pipeline just wait
        if dist.is_initialized():
            dist.barrier()
        return
    
    print(f"\n[Rank {dist.get_rank()}] Saving pipeline checkpoint at iteration {iteration}...")
    
    # ========== Collect checkpoint data ==========
    checkpoint = {
        'iteration': iteration,
        'dataset_position': dataset_position,
        'args': args,
    }
    
    # Save student stage
    checkpoint['student_stage'] = student_stage.state_dict()
    
    # Save teacher stage
    checkpoint['teacher_stage'] = teacher_stage.state_dict()
    
    # Save prototype bank (only exists on last stage)
    if prototype_bank is not None:
        checkpoint['prototype_bank'] = prototype_bank.state_dict()
    
    # Save optimizers
    checkpoint['optimizer_student'] = optimizer_student.state_dict()
    checkpoint['optimizer_prototypes'] = optimizer_prototypes.state_dict()
    
    # Save scaler
    if fp16_scaler is not None:
        checkpoint['fp16_scaler'] = fp16_scaler.state_dict()
    
    # Save loss modules (only on last stage where they're used)
    if prototype_bank is not None:  # Last stage indicator
        checkpoint['loss_modules'] = {
            key: module.state_dict() 
            for key, module in loss_modules.items()
        }
    
    # Save RNG states
    checkpoint['torch_rng_state'] = torch.get_rng_state()
    checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
    checkpoint['numpy_rng_state'] = np.random.get_state()
    checkpoint['random_rng_state'] = random.getstate()
    
    # ========== Save to disk ==========
    checkpoint_path = os.path.join(output_dir, f'checkpoint_iter_{iteration:08d}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest checkpoint
    latest_path = os.path.join(output_dir, 'checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    print(f"[Rank {dist.get_rank()}] ✓ Saved checkpoint: {checkpoint_path}")
    
    # Barrier to ensure all saves complete
    if dist.is_initialized():
        dist.barrier()


def load_pipeline_checkpoint(
    checkpoint_path,
    student_stage,
    teacher_stage,
    prototype_bank,
    optimizer_student,
    optimizer_prototypes,
    fp16_scaler,
    loss_modules,
    local_rank,
    pipeline_group,
    data_group,
):
    """
    Load pipeline checkpoint with proper distributed coordination.
    
    Strategy:
    1. Rank 0 loads checkpoint from disk
    2. Each rank loads its own stage state
    3. Restore RNG states
    
    Returns:
        run_variables: Dict with 'iteration' and 'dataset_position'
    """
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return {'iteration': 0, 'dataset_position': 0}
    
    print(f"\n[Rank {dist.get_rank()}] Loading pipeline checkpoint from {checkpoint_path}...")
    
    # All ranks load (file system can handle it)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # ========== Load model states ==========
    
    # Load student stage
    if 'student_stage' in checkpoint:
        student_stage.load_state_dict(checkpoint['student_stage'], strict=False)
        print(f"[Rank {dist.get_rank()}] Loaded student_stage")
    
    # Load teacher stage
    if 'teacher_stage' in checkpoint:
        teacher_stage.load_state_dict(checkpoint['teacher_stage'], strict=False)
        print(f"[Rank {dist.get_rank()}] Loaded teacher_stage")
    
    # Load prototype bank (only on last stage)
    if prototype_bank is not None and 'prototype_bank' in checkpoint:
        prototype_bank.load_state_dict(checkpoint['prototype_bank'], strict=False)
        print(f"[Rank {dist.get_rank()}] Loaded prototype_bank")
    
    # Load optimizers
    if 'optimizer_student' in checkpoint:
        optimizer_student.load_state_dict(checkpoint['optimizer_student'])
        print(f"[Rank {dist.get_rank()}] Loaded optimizer_student")
    
    if 'optimizer_prototypes' in checkpoint:
        optimizer_prototypes.load_state_dict(checkpoint['optimizer_prototypes'])
        print(f"[Rank {dist.get_rank()}] Loaded optimizer_prototypes")
    
    # Load scaler
    if fp16_scaler is not None and 'fp16_scaler' in checkpoint:
        fp16_scaler.load_state_dict(checkpoint['fp16_scaler'])
        print(f"[Rank {dist.get_rank()}] Loaded fp16_scaler")
    
    # Load loss modules (only on last stage)
    if prototype_bank is not None and 'loss_modules' in checkpoint:
        for key, state_dict in checkpoint['loss_modules'].items():
            if key in loss_modules:
                loss_modules[key].load_state_dict(state_dict)
        print(f"[Rank {dist.get_rank()}] Loaded loss_modules")
    
    # ========== Restore RNG states ==========
    if 'torch_rng_state' in checkpoint:
        try:
            torch.set_rng_state(checkpoint['torch_rng_state'])
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
            np.random.set_state(checkpoint['numpy_rng_state'])
            random.setstate(checkpoint['random_rng_state'])
            print(f"[Rank {dist.get_rank()}] Restored RNG states")
        except Exception as e:
            print(f"[Rank {dist.get_rank()}] WARNING: Failed to restore RNG states: {e}")
    
    # ========== Return run variables ==========
    run_variables = {
        'iteration': checkpoint.get('iteration', 0),
        'dataset_position': checkpoint.get('dataset_position', 0),
    }
    
    print(f"[Rank {dist.get_rank()}] ✓ Loaded checkpoint from iteration {run_variables['iteration']}")
    
    return run_variables



def pipeline_forward_pass(
    stage,
    x,
    local_rank,
    num_stages,
    global_rank,
    gpus_per_node,
    token_masks=None,
):
    """
    Execute forward pass through pipeline stage with send/recv.
    
    Args:
        stage: PipelineStageWrapper for this GPU
        x: Input (from previous stage or data loader)
        local_rank: Local GPU rank within node
        num_stages: Total pipeline stages
        global_rank: Global rank
        gpus_per_node: GPUs per node
        token_masks: Optional masking
    
    Returns:
        Output from this stage (or None if not last stage)
    """
    # Compute on this stage
    output = stage(x, token_masks=token_masks)
    
    # Send to next stage if not last
    if local_rank < num_stages - 1:
        next_rank = global_rank + 1
        
        if isinstance(output, list):
            # Multi-crop case
            for i, tensor in enumerate(output):
                send_tensor(tensor, next_rank, tag=i)
        else:
            send_tensor(output, next_rank, tag=0)
        
        return None  # Not last stage
    else:
        # Last stage: return output
        return output


def pipeline_backward_pass(
    stage,
    grad_output,
    local_rank,
    num_stages,
    global_rank,
    gpus_per_node,
):
    """
    Execute backward pass through pipeline stage.
    
    Args:
        stage: PipelineStageWrapper
        grad_output: Gradients from next stage (or loss)
        local_rank: Local GPU rank
        num_stages: Total stages
        global_rank: Global rank
        gpus_per_node: GPUs per node
    """
    if local_rank == num_stages - 1:
        # Last stage: backward from loss
        grad_output.backward()
    else:
        # Receive gradients from next stage
        # Then backward
        # This is simplified - full implementation needs gradient flow
        pass
    
    # Send gradients to previous stage if not first
    if local_rank > 0:
        # Send gradient to previous stage
        prev_rank = global_rank - 1
        # Implementation depends on activation saving strategy
        pass


def init_distributed_mode_condor(args):
    if not hasattr(args, 'rank_offset'):
        args.rank_offset = 0

    args.world_size = int(os.environ.get('WORLD_SIZE', '1'))

    if 'RANK' in os.environ and 'LOCAL_RANK' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif torch.cuda.is_available():
        print('Running in non-distributed mode on a single GPU.')
        args.rank = 0
        args.gpu = 0
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    # Apply rank offset
    args.rank = args.rank + args.rank_offset

    # Ensure correct GPU assignment
    args.gpu = args.rank % torch.cuda.device_count()

    # Set up the process group
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = args.master_addr
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(args.master_port)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print(f'| distributed init (rank {args.rank}): env://', flush=True)
    print(f'| world_size: {args.world_size}, gpu: {args.gpu}, rank: {args.rank}', flush=True)

    dist.barrier()
    setup_for_distributed(args.rank == 0)



def init_distributed_mode_condor_vanilla(args):
    # Ensure rank_offset is present in args
    if not hasattr(args, 'rank_offset'):
        args.rank_offset = 0

    # Set world_size to 8 (total GPUs across both jobs)
    #args.world_size = 8

    # Set up for Condor environment
    if 'RANK' in os.environ and 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ["RANK"])
    elif torch.cuda.is_available():
        print('Running in non-distributed mode on a single GPU.')
        local_rank = 0
        args.rank = 0
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    # Apply rank offset
    args.rank = args.rank + (args.rank_offset * 4)

    # Ensure correct GPU assignment
    args.gpu = args.rank % torch.cuda.device_count()

    # Set up the process group
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print(f'| distributed init (rank {args.rank}): env://', flush=True)
    print(f'| world_size: {args.world_size}, gpu: {args.gpu}, rank: {args.rank}', flush=True)

    dist.barrier()
    setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


class PCA():
    """
    Class to  compute and apply PCA.
    """
    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """
        Takes a covariance matrix (np.ndarray) as input.
        """
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = np.argsort(d)[::-1][:self.dim]
        d = d[idx]
        v = v[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # for the whitening
        d = np.diag(1. / d**self.whit)

        # principal components
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        # input is from numpy
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T

        # input is from torch and is on GPU
        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(torch.cuda.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)

        # input if from torch, on CPU
        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)
        return torch.mm(torch.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def multi_scale(samples, model):
    v = None
    for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
        feats = model(inp).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return v
