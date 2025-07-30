###############################################################################
# Copyright (c) 2025, Yoonsung Choi
# This file is part of OpenMLXModel.
#
# OpenMLXModel is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# OpenMLXModel is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with OpenMLXModel. If not, see <https://www.gnu.org/licenses/>.
###############################################################################
import mlx.core as mx
import mlx.nn as nn
from common import Linear

class GeGLUFFN(nn.Module):
    """
    GeGLU (GeLU-Gated Linear Unit) FFN.
    FFN: Linear(x) * gelu(Linear(x))
    """
    def __init__(self, d, f, use_bias, dtype=mx.float32):
        super().__init__()
        h = int(d * f)
        self.w1 = Linear(d, h, bias=use_bias, dtype=dtype)
        self.w2 = Linear(d, h, bias=use_bias, dtype=dtype)
        self.proj = Linear(h, d, bias=use_bias, dtype=dtype)

    def __call__(self, x):
        return self.proj(self.w1(x) * nn.gelu(self.w2(x)))