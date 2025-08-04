# Gateless Mixture of Experts LLM

해당 예시는 Pretrain 모델을 생성하고, 모델로 부터 텍스트를 생성하는 예시입니다. 학습데이터는 데이터와 데이터가 가지고 있는 Expert Index를 필요로 합니다. 해당 예시에서는 최대 8개의 Expert를 학습시킵니다. 데이터의 라벨에 따라 특정 Expert만 학습되며, Expert를 명확히 할 수 있다는 점이 존재합니다.

프롬프트를 모델에 통과시킬 때 어떤 전문가에 가중치를 둘 것인지 명확히 지정하여야 하는데 추후 Finetune을 통해 프롬프트가 어떤 전문가를 통하게 할 것인지 gate를 학습시킬 필요가 있습니다. 

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
PYTHONPATH=. uv run examples/GatelessMoELLM/train.py --preset custom --train_steps 10000 --kv_heads 16
```

지난 10000번째의 학습에서 이어서 10000번의 학습 진행
```
PYTHONPATH=. uv run examples/GatelessMoELLM/train.py --preset custom --train_steps 10000 --kv_heads 16 --resume ckpt_010000.json
```

## 생성
2번 전문가를 활성화, "행복의 비밀은 "이라는 문장을 프롬프트로 사용
```
PYTHONPATH=. uv run examples/GatelessMoELLM/generate.py --resume moe_ckpts/ckpt_011300.json --prompt "행복의 비밀은 " --expert-ids 2
```

0번 전문가에게 0.7 만큼의 가중치, 2번 전문가에게 0.3만큼의 가중치로 "행복의 비밀은 "이라는 문장을 프롬프트로 사용
```
PYTHONPATH=. uv run examples/GatelessMoELLM/generate.py --resume moe_ckpts/ckpt_001300.json --prompt "행복의 비밀은 " --expert-ids 0 2 --expert-weights 0.7 0.3
```