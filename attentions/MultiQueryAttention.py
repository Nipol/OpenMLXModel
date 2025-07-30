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
from common import (
    Linear,
    apply_rope,
)

class MultiQueryAttentionTrain(nn.Module):
    """
    Multi-Query Attention(MQA) - 학습 전용.
    모든 쿼리 헤드가 하나의 키/값 헤드를 공유(n_kv=1).
    대형 LLM의 메모리/속도 최적화에 자주 사용됨.

    입력:
        x (mx.array): (B, S, d)
        mask (mx.array): (B, S, S)

    반환:
        mx.array: (B, S, d)

    예시:
        attn = MQALMTrain(d=4096, n_h=32, rope=rope, use_bias=False)
        mask = make_causal_mask(S)
        out = attn(x, mask)
    """
    def __init__(self, d, n_h, rope, use_bias=False, dtype=mx.float32):
        super().__init__()
        self.n_h, self.hd = n_h, d // n_h
        self.scale = self.hd ** -0.5
        self.rope = rope
        self.q = Linear(d, d, bias=use_bias, dtype=dtype)
        self.k = Linear(d, self.hd, bias=use_bias, dtype=dtype)  # 1개만
        self.v = Linear(d, self.hd, bias=use_bias, dtype=dtype)
        self.o = Linear(d, d, bias=use_bias, dtype=dtype)

    def _split(self, x, h):
        b, t, _ = x.shape
        return x.reshape(b, t, h, self.hd).transpose(0,2,1,3)

    def __call__(self, x, mask):
        q = self._split(self.q(x), self.n_h)        # (B, n_h, S, hd)
        k = self._split(self.k(x), 1)               # (B, 1, S, hd)
        v = self._split(self.v(x), 1)
        freqs = self.rope[:q.shape[2]]
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        # MQA: K/V를 쿼리 헤드 개수에 맞게 반복(broadcast)
        k = mx.repeat(k, self.n_h, axis=1)
        v = mx.repeat(v, self.n_h, axis=1)
        y = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=self.scale)
        return self.o(y.transpose(0,2,1,3).reshape(x.shape))
    
class MultiQueryAttentionInfer(nn.Module):
    """
    Multi-Query Attention(MQA) - 추론/Serving 전용.
    모든 쿼리 헤드가 하나의 키/값만 공유(n_kv=1), 캐시도 단일 헤드 기준.

    입력:
        x (mx.array): (B, 1, d)
        cache (dict): {'k': (B,1,S_prev,hd), 'v': ...}
        mask (mx.array): (B, S_full, S_full)

    반환:
        mx.array: (B, 1, d)

    예시:
        attn = MQALMInfer(d=4096, n_h=32, rope=rope, use_bias=False)
        cache = {'k': mx.zeros((B,1,0,hd)), 'v': mx.zeros((B,1,0,hd))}
        out = attn(x, cache, mask)
    """
    def __init__(self, d, n_h, rope, use_bias=False, dtype=mx.float32):
        super().__init__()
        self.n_h, self.hd = n_h, d // n_h
        self.scale = self.hd ** -0.5
        self.rope = rope
        self.q = Linear(d, d, bias=use_bias, dtype=dtype)
        self.k = Linear(d, self.hd, bias=use_bias, dtype=dtype)
        self.v = Linear(d, self.hd, bias=use_bias, dtype=dtype)
        self.o = Linear(d, d, bias=use_bias, dtype=dtype)

    def _split(self, x, h):
        b, t, _ = x.shape
        return x.reshape(b, t, h, self.hd).transpose(0,2,1,3)

    def __call__(self, x, cache, mask):
        q = self._split(self.q(x), self.n_h)         # (B, n_h, 1, hd)
        k_new = self._split(self.k(x), 1)            # (B, 1, 1, hd)
        v_new = self._split(self.v(x), 1)
        freqs = self.rope[:q.shape[2]]
        q, k_new = apply_rope(q, freqs), apply_rope(k_new, freqs)
        k = mx.concatenate([cache['k'], k_new], axis=2)
        v = mx.concatenate([cache['v'], v_new], axis=2)
        cache['k'], cache['v'] = k, v
        # 쿼리 헤드 개수만큼 broadcast
        k = mx.repeat(k, self.n_h, axis=1)
        v = mx.repeat(v, self.n_h, axis=1)
        y = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=self.scale)
        return self.o(y.transpose(0,2,1,3).reshape(x.shape))