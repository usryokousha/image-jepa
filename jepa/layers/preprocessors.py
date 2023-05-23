#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Taken from ImageBind directly

import gzip
import html
import io
import math
from functools import lru_cache
from typing import Callable, List, Optional, Tuple

import ftfy
import numpy as np
import regex as re
import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr
from timm.models.layers import trunc_normal_




