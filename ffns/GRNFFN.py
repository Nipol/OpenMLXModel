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

class GRN(nn.Module):
    """
    Gated Residual Network(GRN).
    - Linear + ELU + Linear로 변환 후,
    - gating branch(Linear+sigmoid)로 게이트를 얻어 곱한다.
    - 입력(skip)을 gating 후 출력에 더함.
    """
    def __init__(self, d, f, use_bias=True, dtype=mx.float32):
        super().__init__()
        h = int(d * f)
        self.fc1 = Linear(d, h, bias=use_bias, dtype=dtype)
        self.fc2 = Linear(h, d, bias=use_bias, dtype=dtype)
        self.gate = Linear(d, d, bias=True, dtype=dtype)
        self.skip = Linear(d, d, bias=False, dtype=dtype)  # skip 연결(원본 input 매핑)

    def __call__(self, x):
        out = self.fc2(nn.elu(self.fc1(x)))
        gate = nn.sigmoid(self.gate(x))
        skip = self.skip(x)
        return out * gate + skip, mx.array(0.0, dtype=x.dtype) # 또는 mx.array(0.0) 등, main_loss와 호환되는 타입
    
    @staticmethod
    def functional_forward(x: mx.array, params: dict) -> mx.array:
        """
        함수형 경로: vmap에서 사용할 staticmethod.
        """
        h = nn.elu(Linear.functional_forward(x, params["fc1"]))
        out = Linear.functional_forward(h, params["fc2"])
        gate_val = nn.sigmoid(Linear.functional_forward(x, params["gate"]))
        skip = Linear.functional_forward(x, params["skip"])
        return out * gate_val + skip