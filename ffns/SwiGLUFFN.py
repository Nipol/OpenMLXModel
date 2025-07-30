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
import math
import mlx.core as mx
import mlx.nn as nn
from common import Linear

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network(FFN) 모듈.

    - 현대 LLM(예: Mistral, LLaMA 등)에서 널리 쓰이는 SwiGLU 구조.
    - 두 Linear(w1, w2) 후 silu(w2(x)) * w1(x) 연산, 이어서 출력 프로젝션(proj).
    - proj 가중치는 작은 분산으로 초기화(2/sqrt(2*num_layers)), 초기 학습 안정성을 높임.
    - 일반적인 Transformer FFN(factor f, hidden dim h)과 동일하게 동작하며,
      실전 대형 언어모델에서 표준적인 사용성을 갖는다.

    Args:
        d (int): 입력/출력 차원(모델 임베딩 차원)
        f (float): hidden layer width factor(보통 4.0~8.0)
        use_bias (bool): Linear 레이어 bias 사용 여부
        num_layers (int): 전체 Transformer 레이어 수 (proj 초기화에 사용)
        dtype: 파라미터 데이터 타입

    입력:
        x (mx.array): (B, S, d) 입력 시퀀스

    반환:
        mx.array: (B, S, d) 출력 시퀀스

    사용 예시:
    -------
    ffn = SwiGLUFFN(d=4096, f=8.0, use_bias=False, num_layers=32)
    y = ffn(x)  # x: (B, S, 4096)
    """
    def __init__(self, d, f, use_bias, num_layers, dtype=mx.float32):
        super().__init__()
        h = int(d * f)
        self.w1 = Linear(d, h, bias=use_bias, dtype=dtype)
        self.w2 = Linear(d, h, bias=use_bias, dtype=dtype)
        self.proj = Linear(h, d, bias=use_bias, dtype=dtype)
        std = 2.0 / math.sqrt(2 * num_layers)
        self.proj.w = mx.random.normal(shape=self.proj.w.shape, scale=std, dtype=dtype)

    def __call__(self, x):
        return self.proj(nn.silu(self.w2(x)) * self.w1(x))