# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# python


from collections import OrderedDict
import torch
import time
from os.path import join
import os
import getpass
import tempfile
import numpy as np
import random

def load_check(checkpoint, normalize_input: bool, normalize_value: bool):
    extras = OrderedDict()
    if normalize_value and 'value_mean_std.running_mean' not in checkpoint['model'].keys():
        extras['value_mean_std.running_mean'] = checkpoint['reward_mean_std']['running_mean']
        extras['value_mean_std.running_var'] = checkpoint['reward_mean_std']['running_var']
        extras['value_mean_std.count'] = checkpoint['reward_mean_std']['count']

    if normalize_input and 'running_mean_std.running_mean' not in checkpoint['model'].keys():
        extras['running_mean_std.running_mean'] = checkpoint['running_mean_std']['running_mean']
        extras['running_mean_std.running_var'] = checkpoint['running_mean_std']['running_var']
        extras['running_mean_std.count'] = checkpoint['running_mean_std']['count']
    
    extras.update(checkpoint['model'])
    checkpoint['model'] = extras
    return checkpoint


def safe_filesystem_op(func, *args, **kwargs):
    """
    This is to prevent spurious crashes related to saving checkpoints or restoring from checkpoints in a Network
    Filesystem environment (i.e. NGC cloud or SLURM)
    """
    num_attempts = 5
    for attempt in range(num_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            print(f'Exception {exc} when trying to execute {func} with args:{args} and kwargs:{kwargs}...')
            wait_sec = 2 ** attempt
            print(f'Waiting {wait_sec} before trying again...')
            time.sleep(wait_sec)

    raise RuntimeError(f'Could not execute {func}, give up after {num_attempts} attempts...')



def safe_load(filename, device=None):
    if device is not None:
        return safe_filesystem_op(torch.load, filename, map_location=device)
    else:
        return safe_filesystem_op(torch.load, filename)

def load_checkpoint(filename, device=None):
    print("=> loading checkpoint '{}'".format(filename))
    state = safe_load(filename, device=device)
    return state


def flatten_dict(d, prefix='', separator='.'):
    res = dict()
    for key, value in d.items():
        if isinstance(value, (dict, OrderedDict)):
            res.update(flatten_dict(value, prefix + key + separator, separator))
        else:
            res[prefix + key] = value

    return res


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def safe_ensure_dir_exists(path):
    """Should be safer in multi-treaded environment."""
    try:
        return ensure_dir_exists(path)
    except FileExistsError:
        return path


def get_username():
    uid = os.getuid()
    try:
        return getpass.getuser()
    except KeyError:
        # worst case scenario - let's just use uid
        return str(uid)


def project_tmp_dir():
    tmp_dir_name = f'ige_{get_username()}'
    return safe_ensure_dir_exists(join(tempfile.gettempdir(), tmp_dir_name))


def set_np_formatting():
    """ formats numpy print """
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def set_seed(seed, torch_deterministic=False, rank=0):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


# EOF
