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
from mlx.utils import tree_map
import argparse, pathlib, json, sys

from attentions.LatentAttention import LatentAttentionInfer
from embedding import Embedding
from ffns.GRNFFN import GRN
from common import (
    Linear,
    RMSNorm,
    LayerScale,
    rope_cache,
    causal_mask,
    one_hot
)

from examples.GatelessMoELLM.train import (
    LLM_CONFIG,
    load_encoding
)

class MOELayer(nn.Module):
    """
    라우팅 행렬을 직접 입력받는 MoE 레이어.
    """
    def __init__(self, d: int, f: float, use_bias: bool, num_experts: int, dtype: mx.Dtype):
        super().__init__()
        self.num_experts = num_experts
        self.experts = [GRN(d, f, use_bias, dtype=dtype) for _ in range(num_experts)]

    def __call__(self, x: mx.array, routing_matrix: mx.array):
        """
        x: 입력 텐서 (B, S, D)
        routing_matrix: (B, S, E) shape의 라우팅 행렬.
        """
        B, S, D = x.shape
        x_reshaped = x.reshape(-1, D)
        routing_matrix_flat = routing_matrix.reshape(-1, self.num_experts)

        # vmap을 사용하여 모든 전문가를 병렬로 실행합니다.
        # 이 단계는 학습 시와 동일한 로직을 사용합니다.
        stacked_expert_params = tree_map(lambda *p: mx.stack(p, axis=0), *[exp.parameters() for exp in self.experts])
        
        vmapped_expert_fn = mx.vmap(
            GRN.functional_forward,
            in_axes=(None, 0),
            out_axes=0
        )
        all_expert_outputs = vmapped_expert_fn(x_reshaped, stacked_expert_params)

        # einsum을 사용하여 라우팅 행렬에 따라 전문가 출력을 가중 합산합니다.
        final_flat_output = mx.einsum('end,ne->nd', all_expert_outputs, routing_matrix_flat)

        return final_flat_output.reshape(B, S, D)

class InferBlock(nn.Module):
    """추론용 Block. 라우팅 행렬을 받아서 MOELayer에 전달합니다."""
    def __init__(self, d, nh, nkv, r, f, use_bias, num_experts=1, dtype=mx.bfloat16):
        super().__init__()
        self.n1 = RMSNorm(d, dtype=dtype)
        self.R1 = LayerScale(d, dtype=dtype)
        self.n2 = RMSNorm(d, dtype=dtype)
        self.R2 = LayerScale(d, dtype=dtype)
        self.attn = LatentAttentionInfer(d, nh, nkv, r, use_bias, dtype=dtype)
        self.ffn = MOELayer(d, f, use_bias, num_experts, dtype=dtype)

    def __call__(self, x: mx.array, routing_matrix: mx.array, c, m: mx.array):
        x = x + self.R1(self.attn(self.n1(x), c, m))
        normed_x = self.n2(x)
        ffn_output = self.ffn(normed_x, routing_matrix)
        x = x + self.R2(ffn_output)
        return x

class InferLLMMini(nn.Module):
    """추론용 전체 모델. 라우팅 행렬을 받아서 각 Block에 전달합니다."""
    def __init__(self, vocab, seq, d, layers, heads, kv_heads, f=4, use_bias=True, dtype=mx.bfloat16, num_experts=1):
        super().__init__()
        self.dtype = dtype
        self.embed = Embedding(vocab, d, dtype=dtype)
        rope = rope_cache(seq, d // heads)
        self.blocks = [InferBlock(d, heads, kv_heads, rope, f, use_bias, num_experts, dtype=dtype) for _ in range(layers)]
        self.norm = RMSNorm(d, dtype=dtype)
        self.lm_head = Linear(d, vocab, bias=False, dtype=dtype)
        self.lm_head.weight = self.embed.tok.weight
        self.cache = [{} for _ in range(layers)]

    def __call__(self, ids: mx.array, routing_matrix: mx.array, m: mx.array = None):
        x, _ = self.embed(ids)
        for i, b in enumerate(self.blocks):
            x = b(x, routing_matrix=routing_matrix, c=self.cache[i], m=m)
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
    model: InferLLMMini, # 임포트한 모델 클래스 이름과 일치시켜야 함
    tokenizer,
    prompt: str,
    expert_ids: list[int],
    expert_weights: list[float],
    max_tokens: int = 100,
    temp: float = 0.8,
    top_p: float = 0.9,
):
    """
    프롬프트를 기반으로 텍스트를 생성합니다. (KV 캐시 적용 버전)
    """
    expert_info = " | ".join([f"Expert {eid}: {w:.2f}" for eid, w in zip(expert_ids, expert_weights)])
    print(f"\n[Using Expert Mix: {expert_info}]\n", flush=True)
    
    print(f"Prompt: {prompt}", end="", flush=True)

    # 1. 라우팅 행렬 생성
    num_experts = model.blocks[0].ffn.num_experts
    # (E,) shape의 기본 라우팅 벡터 생성
    base_routing_vector = mx.zeros((num_experts,), dtype=model.dtype)
    # 지정된 위치에 가중치 설정. MLX는 인덱싱을 통한 업데이트를 지원.
    # 이 연산을 위해 mx.array로 변환 필요
    mx_expert_ids = mx.array(expert_ids)
    mx_expert_weights = mx.array(expert_weights, dtype=model.dtype)
    
    # MLX에서 scatter 업데이트는 직접 지원하지 않으므로, one-hot을 활용하여 만듭니다.
    routing_vector = mx.sum(one_hot(mx_expert_ids, num_experts) * mx_expert_weights[:, None], axis=0)
    
    # 프롬프트를 토큰화하고 모델에 입력 (Prefill 단계)
    prompt_tokens = mx.array([tokenizer.encode(prompt)])
    B, S = prompt_tokens.shape

    # 프롬프트를 위한 라우팅 행렬 브로드캐스팅
    # (E,) -> (1, 1, E) -> (B, S, E)
    routing_matrix = mx.broadcast_to(routing_vector.reshape(1, 1, -1), (B, S, num_experts))

    # 단일 토큰을 위한 라우팅 행렬 (B=1, S=1, E)
    single_token_routing_matrix = routing_vector.reshape(1, 1, -1)

    # 프롬프트 길이만큼 마스크 생성
    mask = causal_mask(S, dtype=mx.bfloat16)
    
    # 맨 처음에는 전체 프롬프트에 대한 logits과 KV 캐시를 생성합니다.
    # 이 단계에서 KV 캐시는 모든 블록에 대해 채워집니다.
    y = model(prompt_tokens, single_token_routing_matrix, m=mask)
    
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
        y = model(next_token.reshape(1, 1), single_token_routing_matrix, m=mask)
        
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

# PYTHONPATH=. uv run examples/GatelessMoELLM/generate.py --resume moe_ckpts/ckpt_001300.json --prompt "행복의 비밀은 " --expert-ids 2

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
    parser.add_argument(
        "--expert-ids",
        type=int,
        nargs='+', # 여러 값을 받을 수 있도록 설정
        required=True,
        help="Space-separated list of expert IDs to use (e.g., 0 2 7)."
    )
    parser.add_argument(
        "--expert-weights",
        type=float,
        nargs='+',
        help="Space-separated list of weights for the experts. Must match the number of expert-ids. Defaults to uniform weights if not provided."
    )
    args = parser.parse_args()

    # --- 가중치 유효성 검사 ---
    expert_ids = args.expert_ids
    expert_weights = args.expert_weights

    if expert_weights is None:
        # 가중치가 제공되지 않으면 균등하게 분배
        num = len(expert_ids)
        expert_weights = [1.0 / num] * num
        print(f"No weights provided, using uniform weights: {expert_weights}")
    elif len(expert_ids) != len(expert_weights):
        sys.exit("Error: The number of --expert-ids and --expert-weights must be the same.")
    
    # 가중치 합이 1에 가깝도록 정규화 (선택 사항이지만 권장)
    total_weight = sum(expert_weights)
    if abs(total_weight - 1.0) > 1e-6:
        print(f"Warning: Expert weights sum to {total_weight}. Normalizing to 1.0.")
        expert_weights = [w / total_weight for w in expert_weights]

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
    model = InferLLMMini(
        vocab_size,
        cfg["block_size"],
        cfg["embd_dim"],
        cfg["num_layers"],
        cfg["num_heads"],
        kv_heads,
        use_bias=cfg["use_bias"],
        dtype = mx.bfloat16,
        num_experts = cfg["num_experts"],
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
        expert_ids=expert_ids,
        expert_weights=expert_weights,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
    )