import mlx.core as mx
import tiktoken
import argparse
import pathlib
import json
import math

from train import dLLMMini, LLM_CONFIG, load_encoding, create_no_effect_mask

def load_model(ckpt_path: pathlib.Path, dtype=mx.bfloat16):
    """
    체크포인트 경로로부터 dLLM 모델을 로드하고 초기화합니다.
    """
    if not ckpt_path.exists() or not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint metadata not found at {ckpt_path}")

    with open(ckpt_path, "r") as f:
        meta = json.load(f)
    
    preset = meta.get("preset", "custom")
    enc_name = meta.get("enc", "o200k_base")
    kv_heads = meta.get("kv_heads")

    if preset not in LLM_CONFIG:
        raise ValueError(f"Preset '{preset}' not found in LLM_CONFIG")

    cfg = LLM_CONFIG[preset]
    enc, vocab_size, mask_token_id = load_encoding(enc_name)
    
    model = dLLMMini(
        vocab_size, 
        cfg["block_size"], 
        cfg["embd_dim"], 
        cfg["num_layers"], 
        cfg["num_heads"], 
        kv_heads, 
        use_bias=cfg["use_bias"],
        dtype=dtype,
    )
    
    # 가중치 파일 경로를 구성합니다.
    step = meta.get("step", 0)
    weights_path = ckpt_path.with_name(f"ckpt_{step:06d}_weights.safetensors")
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weight file not found at {weights_path}")

    print(f"Loading model weights from {weights_path}...")
    model.load_weights(str(weights_path))
    model.eval() # 평가 모드로 전환 (dropout 등 비활성화)
    mx.eval(model.parameters())
    
    return model, enc, cfg, mask_token_id

def gumbel_max_sample(logits, temperature=1.0):
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    gumbel_noise = -mx.log(-mx.log(mx.random.uniform(shape=logits.shape, dtype=mx.float32)))
    return mx.argmax(logits + gumbel_noise * temperature, axis=-1)

def apply_repetition_penalty(logits: mx.array, generated_tokens: mx.array, penalty: float) -> mx.array:
    """
    최근 생성된 토큰에 페널티를 적용하여 반복을 억제합니다.

    - 로짓 값이 양수이면 penalty로 나눕니다. (확률 감소)
    - 로짓 값이 음수이면 penalty를 곱합니다. (확률 감소)
    이는 토큰이 다시 샘플링될 확률을 일관되게 낮추는 효과가 있습니다.

    Args:
        logits (mx.array): 모델의 원본 로짓. Shape: (batch, seq_len, vocab_size).
        generated_tokens (mx.array): 이미 확정된 토큰들의 1D 배열.
        penalty (float): 페널티 계수. 1.0보다 커야 효과가 있습니다 (예: 1.2).

    Returns:
        mx.array: 페널티가 적용된 로짓.
    """
    # 1. 페널티가 1.0이거나, 생성된 토큰이 없으면 아무것도 하지 않고 즉시 반환
    if penalty == 1.0 or generated_tokens.size == 0:
        return logits

    # 2. 페널티를 적용할 고유한 토큰 ID 목록을 만듭니다.
    #    set을 사용하는 것이 중복을 제거하는 가장 간단한 방법입니다.
    unique_tokens = mx.array(list(set(generated_tokens.tolist())))
    
    # 3. 로짓의 어휘집 차원(마지막 차원)을 가져옵니다.
    vocab_size = logits.shape[-1]
    
    # 4. 페널티를 적용할 위치를 나타내는 boolean 마스크를 생성합니다.
    #    shape: (vocab_size,) 이며, unique_tokens에 해당하는 위치만 True가 됩니다.
    #    (mx.arange(vocab_size) == unique_tokens[:, None])는 shape (num_unique, vocab_size)를 생성하고,
    #    any(axis=0)을 통해 이를 1D 마스크로 압축합니다.
    penalize_mask = mx.any(mx.arange(vocab_size) == unique_tokens[:, None], axis=0)

    # 5. 로짓 전체에 대해 페널티 계산을 미리 수행합니다.
    #    이것은 실제로는 아직 적용되지 않은 "만약 페널티를 준다면"에 해당하는 값입니다.
    penalized_logits = mx.where(logits > 0, logits / penalty, logits * penalty)
    
    # 6. 최종 로짓을 결정합니다.
    #    - penalize_mask가 True인 위치(즉, 반복된 토큰)에는 'penalized_logits' 값을 사용하고,
    #    - penalize_mask가 False인 위치에는 원본 'logits' 값을 그대로 사용합니다.
    #    이 과정은 MLX의 브로드캐스팅을 통해 모든 배치와 시퀀스 길이에 대해 동시에 효율적으로 적용됩니다.
    final_logits = mx.where(penalize_mask, penalized_logits, logits)
    
    return final_logits

def top_k_gumbel_sample(logits, k, temperature=1.0):
    """
    Top-k 필터링 후 Gumbel-Max 샘플링을 수행합니다.
    """
    # logits shape: (batch, seq_len, vocab_size)
    
    # 1. 상위 k개의 로짓 값을 찾습니다. mx.partition이 효율적입니다.
    #    k-1을 사용하는 이유는 인덱스가 0부터 시작하기 때문입니다.
    kth_vals = -mx.partition(-logits, kth=k-1, axis=-1)[..., k-1:k]
    
    # 2. 상위 k개에 속하지 않는 로짓 값들을 -inf로 마스킹합니다.
    masked_logits = mx.where(logits < kth_vals, -mx.inf, logits)
    
    # 3. Gumbel-Max 샘플링을 마스킹된 로짓에 적용합니다.
    if temperature == 0:
        return mx.argmax(masked_logits, axis=-1)
        
    gumbel_noise = -mx.log(-mx.log(mx.random.uniform(shape=masked_logits.shape, dtype=mx.float32)))
    return mx.argmax(masked_logits + gumbel_noise * temperature, axis=-1)

def get_num_transfer_tokens(mask_count: int, steps_left: int) -> int:
    """이번 스텝에서 확정할 최대 토큰 수를 계산합니다."""
    return math.ceil(mask_count / steps_left) if steps_left > 0 else mask_count

def generate(
    model: dLLMMini,
    encoder: tiktoken.Encoding,
    prompt: str,
    cfg, # 모델 설정을 직접 받도록 수정
    repetition_penalty: float,
    max_tokens: int = 100,
    denoising_steps: int = 12,
    temperature: float = 1.0,
    confidence_threshold: float = 0.9, # 신뢰도 임계값 추가
    top_k: int = 50,
    mask_token_id: int = 200019,
):
    """
    dLLM을 사용하여 텍스트를 생성합니다. (Iterative Denoising 방식)
    """
    print(f"Prompt: {prompt}", end="", flush=True)

    prompt_tokens = mx.array(encoder.encode(prompt))
    prompt_len = len(prompt_tokens)
    block_size = cfg['block_size']
    
    if prompt_len + max_tokens > block_size:
        raise ValueError("Prompt + max_tokens exceeds block_size.")

    # 초기 시퀀스: [프롬프트, MASK, MASK, ..., MASK]
    tokens = mx.full((1, block_size), mask_token_id, dtype=mx.int32)
    tokens[0, :prompt_len] = prompt_tokens
    
    no_effect_mask = create_no_effect_mask(block_size, model.embed.tok.weight.dtype)
    
    output_text = ""

    for i in range(denoising_steps):
        # 현재 마스킹된 위치와 개수 파악
        condition = (tokens[0] == mask_token_id)
        num_masked = condition.sum().item() # True의 개수를 셈
        # condition을 내림차순으로 정렬하는 인덱스를 얻은 후, True 개수만큼만 슬라이싱
        mask_indices = mx.argsort(-condition.astype(mx.int32))[:num_masked]
        num_masked = len(mask_indices)
        
        if num_masked == 0:
            break
            
        # 1. 모델 예측
        logits = model(tokens, no_effect_mask)

        # 1. '확정된' 토큰의 boolean 마스크를 만듭니다.
        is_confirmed = ~condition
        
        # 2. '확정된' 토큰의 개수를 셉니다.
        num_confirmed = int(is_confirmed.sum().item())
        
        # 3. argsort 트릭을 사용해 '확정된' 토큰의 "정수 인덱스"를 얻습니다.
        #    (True=1, False=0 이므로, 내림차순 정렬하면 True였던 인덱스가 앞으로 옵니다.)
        confirmed_indices = mx.argsort(-is_confirmed.astype(mx.int32))[:num_confirmed]
        
        # 4. 이 정수 인덱스를 사용하여 실제 '확정된' 토큰 ID들을 가져옵니다.
        confirmed_tokens = mx.take(tokens[0], confirmed_indices) # take가 더 명시적입니다.
        
        logits = apply_repetition_penalty(logits, confirmed_tokens, repetition_penalty)
        
        # 2. Gumbel-Max 샘플링으로 전체 시퀀스 예측
        predicted_tokens = top_k_gumbel_sample(logits, k=top_k, temperature=temperature)

        # 3. Confidence-Aware Decoding
        # 3-1. 이번 스텝에 확정할 최대 토큰 수 계산
        steps_left = denoising_steps - i
        num_to_unmask_max = get_num_transfer_tokens(num_masked, steps_left)
        
        # 3-2. 마스킹된 위치에서의 예측 신뢰도 계산
        # mx.take를 사용하여 mask_indices 위치의 값들을 효율적으로 가져옵니다.
        masked_logits = mx.take(logits[0], mask_indices, axis=0)
        masked_predictions = mx.take(predicted_tokens[0], mask_indices, axis=0)
        
        probs = mx.softmax(masked_logits.astype(mx.float32), axis=-1)
        confidence = mx.take_along_axis(probs, masked_predictions[:, None], axis=1).squeeze(1)

        # 3-3. 신뢰도 순으로 정렬
        sorted_confidence_indices = mx.argsort(-confidence)

        # 3-4. 임계값을 넘는 토큰만 확정
        top_k_indices = sorted_confidence_indices[:num_to_unmask_max]
        confident_mask = confidence[top_k_indices] >= confidence_threshold
        
        # 1. 확신하는(True) 토큰이 몇 개인지 셉니다.
        num_confident = int(confident_mask.sum().item())

        # 2. confident_mask에서 True인 값들의 *위치*를 찾습니다.
        #    argsort 트릭을 다시 사용합니다.
        indices_of_true_values = mx.argsort(-confident_mask.astype(mx.int32))[:num_confident]

        # 3. 이 위치 인덱스를 사용하여 top_k_indices에서 최종적으로 확정할 인덱스들을 가져옵니다.
        final_indices_to_unmask_local = top_k_indices[indices_of_true_values]
        
        if len(final_indices_to_unmask_local) == 0 and num_masked > 0:
            final_indices_to_unmask_local = top_k_indices[:1]
        
        # 3-5. 실제 시퀀스에서 업데이트할 위치와 값 결정
        # local 인덱스(0, 1, 2...)를 실제 토큰 시퀀스의 인덱스로 변환합니다.
        positions_to_update = mask_indices[final_indices_to_unmask_local]
        tokens_to_fill = mx.take(predicted_tokens[0], positions_to_update)

        # 4. 시퀀스 업데이트
        if len(positions_to_update) > 0:
            # MLX에서는 인덱싱을 통한 직접적인 할당(in-place update)이 JIT 컴파일과 잘 맞지 않을 수 있습니다.
            # mx.scatter_update를 사용하는 것이 더 안정적일 수 있으나, 여기서는 인덱싱을 그대로 사용합니다.
            # (MLX 0.9.0 이상에서는 대부분의 인덱싱 할당이 잘 동작합니다.)
            tokens[0, positions_to_update] = tokens_to_fill
        
        # 실시간 출력 로직
        current_display_tokens = tokens[0, prompt_len : prompt_len + max_tokens].tolist()
        decoded_part = encoder.decode([t for t in current_display_tokens if t != mask_token_id])
        if len(decoded_part) > len(output_text):
            new_text = decoded_part[len(output_text):]
            print(new_text, end="", flush=True)
            output_text = decoded_part

    print("\n\n[Done]")
    final_tokens = tokens[0, :prompt_len + max_tokens].tolist()
    
    try:
        eos_index = final_tokens.index(encoder.eot_token)
        final_tokens = final_tokens[:eos_index]
    except (ValueError, IndexError):
        pass

    return encoder.decode(final_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from a trained dLLM-mini model.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint .json file.")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="The prompt to start generation from.")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate.")
    parser.add_argument("--denoising-steps", type=int, default=12, help="Number of iterative denoising steps.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. 0 for greedy sampling.")
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=10, help="The number of top tokens to consider for sampling.")
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Penalty for repeating tokens. 1.0 means no penalty.")
    args = parser.parse_args()

    # 모델 로드
    model, encoder, config, mask_token_id = load_model(pathlib.Path(args.ckpt_path))
    
    print(f"Token ID {mask_token_id} is decoded as: '{encoder.decode([mask_token_id])}'")
    
    # 텍스트 생성
    generated_text = generate(
        model, 
        encoder, 
        args.prompt, 
        config,
        max_tokens=args.max_tokens,
        denoising_steps=args.denoising_steps,
        temperature=args.temperature,
        confidence_threshold=args.confidence_threshold,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        mask_token_id=mask_token_id,
    )
    
    print("\n" + "-"*50)
    print("Final Generated Text:")
    print(generated_text)