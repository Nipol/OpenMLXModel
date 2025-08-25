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
from typing import Tuple
from common import (
    Linear,
    RMSNorm,
    apply_rope,
    rope_cache,
)

class LatentAttentionTrain(nn.Module):
    """
    LatentAttention: 학습전용 LoRA·RoPE 기반의 경량 트랜스포머 Attention 모듈.

    - 쿼리(Q)는 LoRA 스타일(압축→Norm→복원) 프로젝션과 RoPE 위치 임베딩이 결합된 두 부분(q_nope, q_pe)으로 분할.
    - 키/값(K/V)도 LoRA 스타일 압축 후, RoPE 임베딩이 적용되는 part(k_pe)와 일반 부분(k_nope, v)로 분리되어 처리.
    - 모든 Q, K, V는 독립적으로 LayerNorm(RMSNorm)을 거치며, position encoding(RoPE)는 head-dim의 절반만 적용.
    - GQA/MQA(그룹·다중 키/값 헤드) 구조를 지원하며, 입력 시퀀스 길이에 따라 RoPE 주파수 테이블을 슬라이싱하여 적용.
    - 어텐션 결과는 output projection과 residual dropout을 거쳐 반환된다.

    Args:
        d (int): 입력 임베딩 차원.
        n_h (int): 쿼리 헤드 수.
        n_kv (int): 키/값 헤드 수 (GQA 지원).
        rope (mx.array): Rotary Position Embedding 주파수 테이블.
        use_bias (bool): 프로젝션 레이어 bias 사용 여부.
        q_lora_rank (int): Q 프로젝션의 LoRA 랭크(압축 차원).
        kv_lora_rank (int): K/V 프로젝션의 LoRA 랭크(압축 차원).
        dropout (float): 드롭아웃 확률.
        dtype: 파라미터 데이터 타입.

    입력:
        x (mx.array): (B, 1, d)  — 배치 B, 1토큰, 임베딩 d.
        mask (mx.array): (B, S_total, S_total) — causal mask(필요시).

    반환:
        mx.array: (B, S, d). LatentAttention을 적용한 출력.

    주요 설명:
      - Q/K/V 프로젝션은 LoRA(Low-rank Adaptation) 스타일로, 파라미터 효율과 계산량 절감을 동시에 추구.
      - RoPE(Rotary Position Embedding)는 Q, K의 head-dim 중 절반에만 적용, 나머지는 'no position' part로 남김.
      - K/V를 위한 공동 압축 및 프로젝션은 파라미터 공유로 메모리와 연산량을 줄임.
      - GQA/MQA 구조를 지원하며, KV 헤드와 Q 헤드가 다를 경우 자동 반복(broadcast) 처리.
      - KV 캐싱/분산 어텐션 등 추가 고급 기능은 추후 확장 가능.

    참고:
      - LatentAttention은 NSA보다 구조가 단순하지만, 파라미터 효율 및 위치 인코딩 제어에 초점을 둔 최신 설계 방식이다.
      - 기존 Multihead Attention보다 훨씬 적은 파라미터와 FLOPs로 유사한 성능을 목표로 한다.
      - RoPE/LoRA의 결합 방식을 쉽게 튜닝하거나, 실험적 구조로 확장하기 용이함.
    """
    def __init__(self, d: int, n_h: int, n_kv: int, rope: mx.array, use_bias: bool,
                 q_lora_rank: int = 64, kv_lora_rank: int = 64, dropout: float = 0.1, dtype = mx.float32):
        super().__init__()
        
        # 기본 설정
        self.num_heads = n_h
        self.num_kv_heads = n_kv
        self.head_dim = d // n_h
        self.scale = self.head_dim ** -0.5
        self.rope = rope

        self.resid_dropout = nn.Dropout(dropout)
        
        self.qk_rope_head_dim = self.head_dim // 2
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = self.head_dim

        # --- 파라미터 정의 ---
        
        # Q 프로젝션 (헤드 수: n_h)
        self.q_a_proj = Linear(d, self.q_lora_rank, bias=use_bias, dtype=dtype)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank)
        self.q_b_proj = Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=use_bias, dtype=dtype)

        # 합쳐진, K, V 공동 압축 및 프로젝션
        self.kv_a_proj_with_rope = Linear(d,
                                          self.kv_lora_rank + self.num_kv_heads * self.qk_rope_head_dim,
                                          bias=use_bias, dtype=dtype)
        
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        # K의 nope 파트와 V는 KV 헤드 수(n_kv)만큼만 생성
        self.kv_b_proj = Linear(self.kv_lora_rank,
                                self.num_kv_heads * (self.qk_nope_head_dim + self.v_head_dim),
                                bias=use_bias, dtype=dtype)
        
        self.o_proj = Linear(self.num_heads * self.v_head_dim, d, bias=use_bias, dtype=dtype)

    def __call__(self, x: mx.array, mask: mx.array):
        batch, seq_len, _ = x.shape

        # 1. Q 프로젝션 (n_h개 헤드)
        q_lora = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_lora).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        # 2. KV 프로젝션 (n_kv개 헤드)
        # 둘을 합친 것임
        kv_a_output = self.kv_a_proj_with_rope(x)

        # split을 사용하여 두 부분을 분리
        kv_lora_compressed, k_pe_unshaped = mx.split(kv_a_output, [self.kv_lora_rank], axis=-1)
        kv_intermediate = self.kv_a_layernorm(kv_lora_compressed)

        kv = self.kv_b_proj(kv_intermediate)
        kv = kv.reshape(batch, seq_len, self.num_kv_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(0, 2, 1, 3)
        k_nope, v = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        k_pe = k_pe_unshaped.reshape(batch, seq_len, self.num_kv_heads, self.qk_rope_head_dim).transpose(0, 2, 1, 3)

        # 3. RoPE 적용
        freqs = self.rope[:seq_len, :self.qk_rope_head_dim]
        q_pe = apply_rope(q_pe, freqs)
        k_pe = apply_rope(k_pe, freqs)
        
        # 4. Q, K 재구성
        query_states = mx.concatenate([q_nope, q_pe], axis=-1)
        key_states = mx.concatenate([k_nope, k_pe], axis=-1)

        # GQA/MQA Style을 위한 헤드 반복
        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            key_states = mx.repeat(key_states, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)

        # 6. 어텐션 계산
        attn_output = mx.fast.scaled_dot_product_attention(
            query_states, 
            key_states, 
            v, 
            scale=self.scale, 
            mask=mask
        )

        # 7. 최종 출력
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        output = self.resid_dropout(self.o_proj(attn_output))
        
        return output

class LatentAttentionInfer(nn.Module):
    """
    Deepseek 2 스타일 Latent Attention의 추론(Serving/Streaming) 전용 모듈.

    - 1-step/스트리밍 추론에 특화: 입력 토큰이 1개일 때 KV 캐시 누적 방식
    - Q, K, V LoRA 스타일 프로젝션 및 RoPE 분할 적용 (논문 구조와 일치)
    - GQA/MQA 모두 지원 (n_kv != n_h일 때 자동 브로드캐스트)
    - 캐시 갱신을 in-place로 지원. 캐시 누락/shape 오류는 바로 드러남

    Args:
        d (int): 임베딩 차원.
        n_h (int): 쿼리 헤드 수.
        n_kv (int): KV 헤드 수 (GQA/MQA).
        rope (mx.array): (max_seq, head_dim//2) RoPE 주파수 테이블.
        use_bias (bool): 프로젝션 레이어 bias 사용 여부.
        q_lora_rank (int): Q LoRA 차원.
        kv_lora_rank (int): KV LoRA 차원.
        dtype: 파라미터 데이터 타입.

    입력:
        x (mx.array): (B, 1, d)  — 배치 B, 1토큰, 임베딩 d.
        cache (dict): {'k': (B, n_kv, S_prev, hd), 'v': ...}  — 이전 K/V 누적
        mask (mx.array): (B, S_total, S_total) — causal mask(필요시).

    반환:
        mx.array: (B, 1, d)  — 어텐션 결과.

    예시:
    -------
    attn = LatentAttentionInfer(d=4096, n_h=32, n_kv=8, rope=rope, use_bias=False)
    cache = {'k': mx.zeros((B,8,0,hd)), 'v': mx.zeros((B,8,0,hd))}
    for t in range(seq_len):
        out = attn(x[:,t:t+1,:], cache, mask)
        # cache['k'], cache['v']는 in-place로 갱신됨

    참고:
      - KV 캐시는 shape/순서 오류시 바로 오류 발생.
      - 학습과는 완전히 분리된 설계.
      - GQA/MQA 모두 자동 반복(broadcast) 처리.
      - RoPE, LayerNorm, LoRA 구조는 학습용과 동일.
    """
    def __init__(self, d: int, n_h: int, n_kv: int, rope: mx.array, use_bias: bool,
                 q_lora_rank: int = 64, kv_lora_rank: int = 64, dtype=mx.float32):
        super().__init__()
        self.num_heads = n_h
        self.num_kv_heads = n_kv
        self.head_dim = d // n_h
        self.scale = self.head_dim ** -0.5
        self.rope = rope # (max_seq_len, qk_rope_head_dim * 2)

        self.qk_rope_head_dim = self.head_dim // 2
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        self.v_head_dim = self.head_dim
        
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.q_a_proj = Linear(d, self.q_lora_rank, bias=use_bias, dtype=dtype)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank)
        self.q_b_proj = Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=use_bias, dtype=dtype)

        self.kv_a_proj_with_rope = Linear(d, self.kv_lora_rank + self.num_kv_heads * self.qk_rope_head_dim, bias=use_bias, dtype=dtype)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = Linear(self.kv_lora_rank, self.num_kv_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=use_bias, dtype=dtype)
        
        self.o_proj = Linear(self.num_heads * self.v_head_dim, d, bias=use_bias, dtype=dtype)

    def __call__(self, x: mx.array, cache: Tuple[mx.array, mx.array], start_pos: int, mask: mx.array = None):
        B, S, _ = x.shape
        key_cache, value_cache = cache

        # print("\n--- [Debug] LatentAttentionInfer.__call__ ---")
        # print(f"Input x shape: (B={B}, S={S}, D={x.shape[-1]})")
        # print(f"Received key_cache shape: {key_cache.shape}")
        # print(f"Received start_pos: {start_pos}")

        # 1. 프로젝션
        q_lora = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_lora).reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        # print(f"  > q (after proj) shape: {q.shape}")
        
        kv_a_output = self.kv_a_proj_with_rope(x)
        kv_lora_compressed, k_pe_unshaped = mx.split(kv_a_output, [self.kv_lora_rank], axis=-1)
        kv_intermediate = self.kv_a_layernorm(kv_lora_compressed)
        kv = self.kv_b_proj(kv_intermediate).reshape(B, S, self.num_kv_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(0, 2, 1, 3)
        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)
        k_pe_unrotated = k_pe_unshaped.reshape(B, S, self.num_kv_heads, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        # print(f"  > values (after proj) shape: {values.shape}")
        # print(f"  > keys (nope part) shape: {k_nope.shape}")
        # print(f"  > keys (pe part) shape: {k_pe_unrotated.shape}")

        # 2. RoPE 적용
        # freqs = self.rope[start_pos : start_pos + S, :]
        q_nope, q_pe_unrotated = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        # self.rope 테이블 (너비: D_head)에서,
        # RoPE가 적용될 부분(너비: qk_rope_head_dim)만큼만 잘라냅니다.
        freqs = self.rope[start_pos : start_pos + S, :self.qk_rope_head_dim]
        # print(f"  > Sliced freqs shape: {freqs.shape}")
        
        q_pe_rotated = apply_rope(q_pe_unrotated, freqs)
        k_pe_rotated = apply_rope(k_pe_unrotated, freqs)
        
        queries = mx.concatenate([q_nope, q_pe_rotated], axis=-1)
        keys = mx.concatenate([k_nope, k_pe_rotated], axis=-1)
        # print(f"  > final queries (after RoPE) shape: {queries.shape}")
        # print(f"  > final keys (after RoPE) shape: {keys.shape}")

        # 3. [★★핵심 수정★★] 캐시 업데이트 (수동 재구성)
        # 캐시의 세 부분으로 나눔: 업데이트 전, 업데이트 될 부분, 업데이트 후
        pre_slice_k = key_cache[:, :, :start_pos, :]
        post_slice_k = key_cache[:, :, start_pos + S:, :]
        
        pre_slice_v = value_cache[:, :, :start_pos, :]
        post_slice_v = value_cache[:, :, start_pos + S:, :]
        
        # 세 부분을 다시 합쳐서 새로운 캐시 텐서 생성
        new_key_cache = mx.concatenate([pre_slice_k, keys, post_slice_k], axis=2)
        new_value_cache = mx.concatenate([pre_slice_v, values, post_slice_v], axis=2)
        # print(f"  > new_key_cache shape: {new_key_cache.shape}")

        # 4. 어텐션 계산 준비
        final_keys_len = start_pos + S
        final_keys = new_key_cache[:, :, :final_keys_len, :]
        final_values = new_value_cache[:, :, :final_keys_len, :]
        # print(f"  > final_keys for attention shape: {final_keys.shape}")
        
        if self.num_kv_heads != self.num_heads:
            final_keys = mx.repeat(final_keys, self.num_heads // self.num_kv_heads, axis=1)
            final_values = mx.repeat(final_values, self.num_heads // self.num_kv_heads, axis=1)

        # 5. 어텐션 계산
        attn_output = mx.fast.scaled_dot_product_attention(queries, final_keys, final_values, scale=self.scale, mask=mask)
        output = self.o_proj(attn_output.transpose(0, 2, 1, 3).reshape(B, S, -1))
        
        # print(f"  > Final output shape: {output.shape}")
        return output, (new_key_cache, new_value_cache)

# --- 로직 검증 테스트 (최종) ---
if __name__ == '__main__':
    B, S_max, D = 1, 128, 64
    H, H_kv = 4, 2
    Q_LORA, KV_LORA = 16, 16
    D_head = D // H
    qk_rope_head_dim = D_head // 2
    DTYPE = mx.float32
    
    print("LatentAttentionInfer 최종 로직 검증 테스트 (Shape 일치)")
    
    # [★★핵심 수정★★] rope_table을 qk_rope_head_dim 차원으로 생성
    rope_table = rope_cache(S_max, qk_rope_head_dim, dtype=DTYPE)
    print(f"생성된 RoPE 테이블 shape: {rope_table.shape}")
    
    model = LatentAttentionInfer(d=D, n_h=H, n_kv=H_kv, rope=rope_table, use_bias=True, q_lora_rank=Q_LORA, kv_lora_rank=KV_LORA, dtype=DTYPE)
    mx.eval(model.parameters())
    
    key_cache = mx.zeros((B, H_kv, S_max, D_head), dtype=DTYPE)
    value_cache = mx.zeros((B, H_kv, S_max, D_head), dtype=DTYPE)
    
    print("\n[Step 1] Prefill 단계 테스트")
    prompt_len = 10
    prompt_tokens = mx.random.normal(shape=(B, prompt_len, D))
    
    # Prefill 시 start_pos=0
    output_prefill, caches_after_prefill = model(prompt_tokens, (key_cache, value_cache), 0)
    mx.eval(output_prefill, caches_after_prefill[0])
    
    print(f"  Prefill 후 Key 캐시 shape: {caches_after_prefill[0].shape}")
    assert mx.sum(caches_after_prefill[0][:, :, :prompt_len, :]).item() != 0
    print("  ✅ Prefill 단계 검증 통과!")

    print("\n[Step 2] Decode 단계 테스트")
    current_caches = caches_after_prefill
    current_pos = prompt_len
    
    single_token = mx.random.normal(shape=(B, 1, D))
    # Decode 시 start_pos=current_pos
    output_decode, current_caches = model(single_token, current_caches, current_pos)
    mx.eval(output_decode, current_caches[0])
    
    print(f"  Decode 후 Key 캐시 shape: {current_caches[0].shape}")
    assert mx.sum(current_caches[0][:, :, current_pos:current_pos+1, :]).item() != 0
    print("  ✅ Decode 단계 검증 통과!")
    
    print("\n🎉 모든 검증을 통과했습니다.")