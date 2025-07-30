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
    
    def _forward(self, params, x):
        # 함수형 스타일
        # params는 self.parameters()와 동일한 구조의 딕셔너리
        h = nn.elu(mx.matmul(x, params["fc1"]["w"].T) + params["fc1"]["b"])
        out = mx.matmul(h, params["fc2"]["w"].T) + params["fc2"]["b"]
        
        gate = nn.sigmoid(mx.matmul(x, params["gate"]["w"].T) + params["gate"]["b"])
        skip = mx.matmul(x, params["skip"]["w"].T)
        
        return out * gate + skip, mx.array(0.0, dtype=x.dtype)