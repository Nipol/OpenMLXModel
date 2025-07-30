###############################################################################
# Copyright (c) 2025, Yoonsung Choi
# This file is part of OpenMLXModel.

# OpenMLXModel is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
###############################################################################
import mlx.core as mx
import mlx.nn as nn
from common import Linear

class ReGLUFFN(nn.Module):
    """
    ReGLU (ReLU-Gated Linear Unit) FFN.
    FFN: Linear(x) * relu(Linear(x))
    """
    def __init__(self, d, f, use_bias, dtype=mx.float32):
        super().__init__()
        h = int(d * f)
        self.w1 = Linear(d, h, bias=use_bias, dtype=dtype)
        self.w2 = Linear(d, h, bias=use_bias, dtype=dtype)
        self.proj = Linear(h, d, bias=use_bias, dtype=dtype)

    def __call__(self, x):
        return self.proj(self.w1(x) * nn.relu(self.w2(x)))