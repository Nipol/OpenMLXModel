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

class GroupQueryAttentionTrain(nn.Module):
    """
    Grouped Query Attention(GQA) - 학습 전용.
    여러 쿼리 헤드가 그룹별로 K/V 헤드를 공유.
    대형 LLM에서 주로 메모리/속도 최적화를 위해 사용.

    Args:
        d (int): 임베딩 차원.
        n_h (int): 쿼리 헤드 수.
        n_kv (int): 키/값 헤드 수(n_kv < n_h).
        rope (mx.array): RoPE 주파수 테이블.
        use_bias (bool): bias 여부.
        dtype: 파라미터 타입.

    입력:
        x (mx.array): (B, S, d)
        mask (mx.array): (B, S, S)

    반환:
        mx.array: (B, S, d)

    예시:
        attn = GQALMTrain(d=4096, n_h=32, n_kv=4, rope=rope, use_bias=False)
        mask = make_causal_mask(S)
        out = attn(x, mask)
    """
    def __init__(self, d, n_h, n_kv, rope, use_bias=False, dtype=mx.float32):
        super().__init__()
        self.n_h, self.n_kv, self.hd = n_h, n_kv, d // n_h
        self.scale = self.hd ** -0.5
        self.rope = rope
        self.q = Linear(d, d, bias=use_bias, dtype=dtype)
        self.k = Linear(d, self.hd * n_kv, bias=use_bias, dtype=dtype)
        self.v = Linear(d, self.hd * n_kv, bias=use_bias, dtype=dtype)
        self.o = Linear(d, d, bias=use_bias, dtype=dtype)

    def _split(self, x, h):
        b, t, _ = x.shape
        return x.reshape(b, t, h, self.hd).transpose(0,2,1,3)

    def __call__(self, x, mask):
        q = self._split(self.q(x), self.n_h)
        k = self._split(self.k(x), self.n_kv)
        v = self._split(self.v(x), self.n_kv)
        freqs = self.rope[:q.shape[2]]
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        # GQA: KV헤드를 쿼리헤드에 맞게 반복(broadcast)
        rep = self.n_h // self.n_kv
        k = mx.repeat(k, rep, axis=1)
        v = mx.repeat(v, rep, axis=1)
        y = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=self.scale)
        return self.o(y.transpose(0,2,1,3).reshape(x.shape))
    
class GroupQueryAttentionInfer(nn.Module):
    """
    Grouped Query Attention(GQA) - 추론/Serving 전용.
    이전 K/V를 캐시에 누적, 쿼리 헤드별로 반복적 broadcast 처리.

    입력:
        x (mx.array): (B, 1, d)
        cache (dict): {'k': (B, n_kv, S_prev, hd), 'v': ...}
        mask (mx.array): (B, S_full, S_full)

    반환:
        mx.array: (B, 1, d)

    예시:
        attn = GQALMInfer(d=4096, n_h=32, n_kv=4, rope=rope, use_bias=False)
        cache = {'k': mx.zeros((B,4,0,hd)), 'v': mx.zeros((B,4,0,hd))}
        out = attn(x, cache, mask)
    """
    def __init__(self, d, n_h, n_kv, rope, use_bias=False, dtype=mx.float32):
        super().__init__()
        self.n_h, self.n_kv, self.hd = n_h, n_kv, d // n_h
        self.scale = self.hd ** -0.5
        self.rope = rope
        self.q = Linear(d, d, bias=use_bias, dtype=dtype)
        self.k = Linear(d, self.hd * n_kv, bias=use_bias, dtype=dtype)
        self.v = Linear(d, self.hd * n_kv, bias=use_bias, dtype=dtype)
        self.o = Linear(d, d, bias=use_bias, dtype=dtype)

    def _split(self, x, h):
        b, t, _ = x.shape
        return x.reshape(b, t, h, self.hd).transpose(0,2,1,3)

    def __call__(self, x, cache, mask):
        q = self._split(self.q(x), self.n_h)
        k_new = self._split(self.k(x), self.n_kv)
        v_new = self._split(self.v(x), self.n_kv)
        freqs = self.rope[:q.shape[2]]
        q, k_new = apply_rope(q, freqs), apply_rope(k_new, freqs)
        k = mx.concatenate([cache['k'], k_new], axis=2)
        v = mx.concatenate([cache['v'], v_new], axis=2)
        cache['k'], cache['v'] = k, v
        rep = self.n_h // self.n_kv
        k = mx.repeat(k, rep, axis=1)
        v = mx.repeat(v, rep, axis=1)
        y = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=self.scale)
        return self.o(y.transpose(0,2,1,3).reshape(x.shape))