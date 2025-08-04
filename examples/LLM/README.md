# Large Language Model

해당 예시는 Pretrain 모델을 생성하고, 모델로 부터 텍스트를 생성하는 예시입니다. 추후 Finetune을 통해 대화형 모델이 될 수 있습니다.

## 학습
- GRN
- Latent Attention
- 16 Layers
- 16 Heads
- 512 Embeddings
- 1024 Blocks
- 5e-6 Learning Rate

16개의 KV헤드로 10000번의 학습 진행
```
PYTHONPATH=. uv run examples/LLM/train.py --preset custom --train_steps 10000 --kv_heads 16
```

지난 10000번째의 학습에서 이어서 10000번의 학습 진행
```
PYTHONPATH=. uv run examples/LLM/train.py --preset custom --train_steps 10000 --kv_heads 16 --resume ckpt_010000.json
```

## 생성
2번 전문가를 활성화, "행복의 비밀은 "이라는 문장을 프롬프트로 사용
```
PYTHONPATH=. uv run examples/LLM/generate.py --resume moe_ckpts/ckpt_011300.json --prompt "행복의 비밀은 " 
```
