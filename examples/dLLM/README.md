# Diffusion based Large Language Model

해당 예시는 Pretrain 모델을 생성하고, 모델로 부터 텍스트를 생성하는 예시입니다. 특히, Diffusion 기반의 LLM은 학습보다도 토큰을 결정하는 방식에 공을 많이 들이고 있으므로, 이를 확인하는 것도 중요합니다.

## 학습
- GRN
- Latent Attention
- 12 Layers
- 16 Heads
- 512 Embeddings
- 1024 Blocks
- 5e-6 Learning Rate

16개의 KV헤드로 10000번의 학습 진행
```
PYTHONPATH=. uv run examples/dLLM/train.py --preset custom --kv_heads 16 --train_steps 10000
```

지난 10000번째의 학습에서 이어서 10000번의 학습 진행
```
PYTHONPATH=. uv run examples/dLLM/train.py --preset custom --kv_heads 16 --train_steps 10000 --resume ckpt_010000.json
```

## 생성
"행복의 비밀은 "이라는 문장을 프롬프트로 사용하여, 최대 20토큰, 최대 디노이징 스텝을 30.
```
PYTHONPATH=. uv run examples/dLLM/generate.py --ckpt-path dllm_ckpts/ckpt_006600.json --prompt "행복의 비밀은 " --max-tokens 20 --denoising-steps 30 --prompt "행복의 비밀은 "
```

같은 방식으로 하되, 토큰의 확신도에 대한 임계값을 0.97로 설정하면서, 온도 또한 0.1로 낮춤
```
PYTHONPATH=. uv run examples/dLLM/generate.py --ckpt-path dllm_ckpts/ckpt_006600.json --prompt "행복의 비밀은 " --max-tokens 20 --denoising-steps 30 --prompt "행복의 비밀은 " --confidence-threshold 0.97 --temperature 0.1
```