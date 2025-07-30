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

class Embedding(nn.Module):
    def __init__(self, vocab, d, dtype = mx.float32):
        super().__init__()
        # nn.Embedding 자체는 dtype을 직접 받지 않으므로, 내부의 weight를 직접 초기화해주는 것이 올바름
        self.tok=nn.Embedding(vocab, d)

        # 1. 작은 표준편차로 명시적 초기화
        # 2. DTYPE을 명시적으로 지정하여 타입 일관성 확보
        std = 1.0 / math.sqrt(d)
        self.tok.weight = mx.random.normal(
            shape=self.tok.weight.shape,
            scale=std,
            dtype=dtype
        )

    def __call__(self,ids):
        x = self.tok(ids)
        return x, None