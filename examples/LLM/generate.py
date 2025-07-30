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
import argparse, pathlib, json
import argparse
import json
import pathlib

from attentions.LatentAttention import LatentAttentionInfer
from ffns.GRNFFN import GRN
from embedding import Embedding
from common import (
    Linear,
    RMSNorm,
    LayerScale,
    rope_cache,
    causal_mask
)

from examples.LLM.train import (
    LLM_CONFIG,
    load_encoding,
)

class Block(nn.Module):
    def __init__(self, d, nh, nkv, r, f, use_bias, dtype = mx.bfloat16):
        super().__init__()
        self.n1 = RMSNorm(d, dtype=dtype)
        self.R1 = LayerScale(d, dtype=dtype)
        self.n2 = RMSNorm(d, dtype=dtype)
        self.R2 = LayerScale(d, dtype=dtype)
        self.attn = LatentAttentionInfer(d, nh, nkv, r, use_bias, dtype = dtype)
        self.ffn = GRN(d, f, use_bias, dtype = dtype)

    def __call__(self, x, c, m):
        x = x + self.R1(self.attn(self.n1(x), c, m))
        normed_x = self.n2(x)
        ffn_output, aux_loss = self.ffn(normed_x)
        x = x + self.R2(ffn_output)
        return x, aux_loss

class LLMMini(nn.Module):
    """
    학습(Train) 전용 LLM
    """
    def __init__(self, vocab, seq, d, layers, heads, kv_heads, f=4, use_bias=True, dtype = mx.bfloat16):
        super().__init__()
        self.embed=Embedding(vocab, d, dtype=dtype)
        rope = rope_cache(seq, d // heads)
        self.blocks=[Block(d, heads, kv_heads, rope, f, use_bias, dtype=dtype) for _ in range(layers)]
        self.norm=RMSNorm(d, dtype = dtype)
        self.lm_head=Linear(d, vocab, bias=False, dtype=dtype)
        self.lm_head.weight=self.embed.tok.weight
        self.cache = [{} for _ in range(layers)]

    def __call__(self, ids, m=None):
        x, _ = self.embed(ids)
        for i, b in enumerate(self.blocks):
            x, _ = b(x, c=self.cache[i], m=m)
        x_norm = self.norm(x)
        logits = self.lm_head(x_norm)
        return logits


def sample(logits: mx.array, temp: float = 1.0, top_p: float = 0.9):
    """
    Top-p (nucleus) 샘플링과 temperature를 적용하여 다음 토큰을 선택합니다.
    """
    # temperature 적용
    if temp > 0:
        logits /= temp
    
    # 소프트맥스를 통해 확률 분포 생성
    probs = mx.softmax(logits, axis=-1)
    
    # top-p (nucleus) 샘플링
    sorted_probs = mx.sort(probs)[::-1]
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    
    # p를 초과하는 확률을 가진 토큰들을 제외할 인덱스를 찾음
    cutoff_index = mx.sum(cumulative_probs < top_p)
    cutoff_value = sorted_probs[cutoff_index]
    
    # cutoff 값보다 작은 확률을 가진 토큰들은 확률을 0으로 설정
    probs = mx.where(probs < cutoff_value, 0.0, probs)
    
    # 확률 재정규화
    probs /= mx.sum(probs)
    
    # 재정규화된 확률 분포에서 샘플링
    return mx.random.categorical(mx.log(probs))

def generate(
    model: LLMMini, # 임포트한 모델 클래스 이름과 일치시켜야 함
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temp: float = 0.8,
    top_p: float = 0.9,
):
    """
    프롬프트를 기반으로 텍스트를 생성합니다. (KV 캐시 적용 버전)
    """
    print(f"Prompt: {prompt}", end="", flush=True)
    
    # 1. 프롬프트를 토큰화하고 모델에 입력 (Prefill 단계)
    prompt_tokens = mx.array([tokenizer.encode(prompt)])
    # 프롬프트 길이만큼 마스크 생성
    seq_len = prompt_tokens.shape[1]
    mask = causal_mask(seq_len, dtype=mx.bfloat16)
    
    # 맨 처음에는 전체 프롬프트에 대한 logits과 KV 캐시를 생성합니다.
    # 이 단계에서 KV 캐시는 모든 블록에 대해 채워집니다.
    y = model(prompt_tokens, m=mask)
    
    # 마지막 토큰의 로짓만 사용하여 첫 번째 새로운 토큰을 샘플링합니다.
    last_token_logits = y[:, -1, :]
    next_token = sample(last_token_logits, temp, top_p)
    
    # 생성된 토큰을 즉시 디코딩하여 출력
    token_text = tokenizer.decode(next_token.tolist())
    print(token_text, end="", flush=True)

    # 2. 자기회귀 생성 루프 (Decode 단계)
    #    - 이제부터는 매번 토큰을 '하나씩만' 모델에 입력합니다.
    #    - KV 캐시는 각 블록의 __call__ 메소드 내에서 자동으로 관리됩니다.
    for i in range(max_tokens - 1):
        # 모델에 이전에 생성된 '단일' 토큰을 입력합니다.
        y = model(next_token.reshape(1, 1), m=mask)
        
        # 모델 출력은 항상 (B, S, V) 형태이므로, S=1인 로짓을 가져옵니다.
        logits = y[:, -1, :]
        
        # 다음 토큰 샘플링
        next_token = sample(logits, temp, top_p)

        # 생성된 토큰을 즉시 디코딩하여 출력
        token_text = tokenizer.decode(next_token.tolist())
        print(token_text, end="", flush=True)
        
        # EOT 토큰이 생성되면 루프 종료
        if next_token.item() == tokenizer.eot_token:
            break

    print("\n\n[Done]")

# PYTHONPATH=. uv run examples/LLM/llmgenerate.py --resume llm_ckpts/ckpt_001000.json --prompt "The secret to happiness is"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from a trained model.")
    parser.add_argument(
        "--resume",
        type=str,
        required=True,
        help="Path to the checkpoint JSON file (e.g., ckpts/ckpt_000100.json).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains.",
        help="The prompt to start generation from.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=200, help="Maximum number of tokens to generate."
    )
    parser.add_argument(
        "--temp", type=float, default=0.8, help="Sampling temperature."
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling."
    )
    args = parser.parse_args()

    # 1. 체크포인트 메타데이터 로드
    ckpt_path = pathlib.Path(args.resume)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {ckpt_path}")

    with open(ckpt_path, "r") as f:
        meta = json.load(f)

    # 2. 모델 설정 및 토크나이저 로드
    preset = meta["preset"]
    enc_name = meta["enc"]
    kv_heads = meta["kv_heads"]
    
    cfg = LLM_CONFIG[preset]
    tokenizer, vocab_size = load_encoding(enc_name)

    # 3. 모델 아키텍처 생성
    model = LLMMini(
        vocab_size,
        cfg["block_size"],
        cfg["embd_dim"],
        cfg["num_layers"],
        cfg["num_heads"],
        kv_heads,
        use_bias=cfg["use_bias"],
    )

    # 4. 가중치 로드
    step = meta.get("step", 0)
    weights_path = ckpt_path.with_name(f"ckpt_{step:06d}_weights.safetensors")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weights_path}")

    print(f"Loading model weights from {weights_path}...")
    model.load_weights(str(weights_path))
    model.eval() # 모델을 평가 모드로 설정 (dropout 등 비활성화)
    mx.eval(model.parameters()) # 모델 파라미터를 실제로 로드

    print("\nModel loaded successfully. Starting generation...")
    print("-" * 50)

    # 5. 텍스트 생성 실행
    generate(
        model,
        tokenizer,
        args.prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
    )