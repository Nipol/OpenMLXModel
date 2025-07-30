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
    sliding_window_causal_mask
)

class MultiHeadAttentionTrain(nn.Module):
    """
    학습 전용 Multi-Head Attention 모듈.
    
    - 현대 LLM(예: GPT-3/4, Llama, Mistral 등) 학습 시나리오에 최적화
    - 입력 전체 시퀀스를 한 번에 처리, 캐시/스트림 사용 없음
    - Causal 마스크(=lower-triangular, 미래 토큰 attend 금지) 적용 필수
    - RoPE 위치 인코딩 적용
    - 배치 학습에 적합하며, 불필요한 if/분기 없이 빠른 연산

    Args:
        d (int): 임베딩 차원
        n_h (int): 쿼리/키/값 헤드 수
        rope (mx.array): (최대 시퀀스, head_dim) RoPE 주파수 테이블
        use_bias (bool): Linear 레이어 bias 사용 여부
        dtype: 파라미터 타입

    입력:
        x (mx.array): (B, S, d), 배치 B, 시퀀스 S, 임베딩 d
        mask (mx.array): (B, S, S), causal/padding 마스크. None일 경우 full attention

    반환:
        mx.array: (B, S, d), 어텐션 결과

    사용 예시:
    -------
    attn = MHALMTrain(d=4096, n_h=32, rope=rope_freqs, use_bias=False)
    mask = make_causal_mask(S)
    out = attn(x, mask=mask)
    """
    def __init__(self, d, n_h, rope, use_bias=False, dtype=mx.float32):
        super().__init__()
        self.n_h, self.hd = n_h, d//n_h
        self.scale = self.hd ** -0.5
        self.rope = rope
        self.q = Linear(d, d, bias=use_bias, dtype=dtype)
        self.k = Linear(d, d, bias=use_bias, dtype=dtype)
        self.v = Linear(d, d, bias=use_bias, dtype=dtype)
        self.o = Linear(d, d, bias=use_bias, dtype=dtype)

    def _split(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.n_h, self.hd).transpose(0,2,1,3)

    def __call__(self, x, mask=None):
        q = self._split(self.q(x))
        k = self._split(self.k(x))
        v = self._split(self.v(x))
        freqs = self.rope[:q.shape[2]]
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        y = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=self.scale)
        return self.o(y.transpose(0,2,1,3).reshape(x.shape))

class MultiHeadAttentionInfer(nn.Module):
    """
    추론(Serving/Autoregressive) 전용 Multi-Head Attention 모듈.

    - 현대 LLM 추론(1-토큰 또는 스트리밍) 시나리오에 맞춤
    - 이전 토큰들의 K/V를 캐시에 누적, 새로운 토큰을 concat하여 처리
    - 항상 Causal 마스크 적용 가능 (외부에서 넘기면 shape 일치만 보장)
    - RoPE 위치 인코딩 적용
    - 분기/불필요 연산 최소화, latency/효율성에 집중

    Args:
        d (int): 임베딩 차원
        n_h (int): 쿼리/키/값 헤드 수
        rope (mx.array): (최대 시퀀스, head_dim) RoPE 주파수 테이블
        use_bias (bool): Linear 레이어 bias 사용 여부
        dtype: 파라미터 타입

    입력:
        x (mx.array): (B, 1, d), 배치 B, 단일 토큰, 임베딩 d
        cache (dict): {'k': (B, n_h, S_prev, hd), 'v': ...}, 이전 K/V 누적 텐서
        mask (mx.array): (B, S_full, S_full) 또는 (B, 1, S_full), causal 마스크

    반환:
        mx.array: (B, 1, d), 어텐션 결과

    사용 예시:
    -------
    attn = MHALMInfer(d=4096, n_h=32, rope=rope_freqs, use_bias=False)
    cache = {'k': mx.zeros((B,32,0,hd)), 'v': mx.zeros((B,32,0,hd))}
    for t in range(seq_len):
        out = attn(x[:,t:t+1,:], cache=cache, mask=mask)
        # cache 내부는 in-place로 계속 업데이트됨
    """
    def __init__(self, d, n_h, rope, use_bias=False, dtype=mx.float32):
        super().__init__()
        self.n_h, self.hd = n_h, d//n_h
        self.scale = self.hd ** -0.5
        self.rope = rope
        self.q = Linear(d, d, bias=use_bias, dtype=dtype)
        self.k = Linear(d, d, bias=use_bias, dtype=dtype)
        self.v = Linear(d, d, bias=use_bias, dtype=dtype)
        self.o = Linear(d, d, bias=use_bias, dtype=dtype)

    def _split(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.n_h, self.hd).transpose(0,2,1,3)

    def __call__(self, x, cache, mask):
        q = self._split(self.q(x))   # (B, n_h, 1, hd)
        k_new = self._split(self.k(x)) # (B, n_h, 1, hd)
        v_new = self._split(self.v(x)) # (B, n_h, 1, hd)
        freqs = self.rope[:, :q.shape[-1]]
        q = apply_rope(q, freqs)
        k_new = apply_rope(k_new, freqs)
        # 캐시 누적 (k/v) → (B, n_h, S_total, hd)
        k = mx.concatenate([cache['k'], k_new], axis=2)
        v = mx.concatenate([cache['v'], v_new], axis=2)
        cache['k'], cache['v'] = k, v  # in-place 갱신
        y = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=self.scale)
        return self.o(y.transpose(0,2,1,3).reshape(x.shape))
