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
import mlx.optimizers as optim
import tiktoken
import numpy as np # only for data Loader
import math, argparse, sys, pathlib, json

from typing import Dict, Tuple
from mlx.utils import tree_unflatten, tree_flatten, tree_map
from attentions.LatentAttention import LatentAttentionTrain
from ffns.GRNFFN import GRN
from embedding import Embedding
from common import (
    Linear,
    RMSNorm,
    LayerScale,
    one_hot,
    rope_cache,
    causal_mask,
    count_params
)

class MOELayer(nn.Module):
    """
    데이터 라벨에 따라 전문가를 명시적으로 선택하는 MoE 레이어.
    """
    def __init__(self, d: int, f: float, use_bias: bool, num_experts: int, dtype: mx.Dtype):
        super().__init__()
        self.num_experts = num_experts
        # 각 라벨에 해당하는 전문가를 생성합니다.
        self.experts = [GRN(d, f, use_bias, dtype=dtype) for _ in range(num_experts)]

    def __call__(self, x: mx.array, labels: mx.array):
        """
        x: 입력 텐서 (B, S, D)
        labels: 각 토큰에 대한 전문가를 지정하는 라벨 텐서 (B, S)
        """
        B, S, D = x.shape
        x_reshaped = x.reshape(-1, D)
        labels_flat = labels.reshape(-1)

        # 1. 모든 토큰을 모든 전문가에게 전달 (vmap)
        stacked_expert_params = tree_map(lambda *p: mx.stack(p, axis=0), *[exp.parameters() for exp in self.experts])
        
        # GRN의 __call__을 직접 참조하되, 파라미터를 명시적으로 전달하는 람다 함수 사용
        vmapped_expert_fn = mx.vmap(
            GRN.functional_forward,
            in_axes=(None, 0),
            out_axes=0
        )
        
        all_expert_outputs = vmapped_expert_fn(x_reshaped, stacked_expert_params)

        # 2. 라벨로부터 라우팅 행렬 생성 (핵심)
        # 라벨을 원-핫 인코딩하여 라우팅 행렬을 직접 만듭니다.
        # 이 행렬의 각 행은 해당 토큰이 가야 할 전문가 위치에만 1을 가집니다.
        routing_matrix = one_hot(labels_flat, self.num_experts).astype(x.dtype)
        
        # 3. 가중 합산 (einsum)
        # routing_matrix에 의해 정확히 라벨에 해당하는 전문가의 출력만 선택됩니다.
        final_flat_output = mx.einsum('end,ne->nd', all_expert_outputs, routing_matrix)

        # 이 방식에서는 MoE의 부하 분산 손실(aux_loss)이 의미가 없습니다.
        # 라우팅이 학습되지 않기 때문입니다.
        return final_flat_output.reshape(B, S, D)

class Block(nn.Module):
    def __init__(self, d, nh, nkv, r, f, use_bias, num_experts=1, dtype = mx.bfloat16):
        super().__init__()
        self.n1 = RMSNorm(d, dtype=dtype)
        self.R1 = LayerScale(d, dtype=dtype)
        self.n2 = RMSNorm(d, dtype=dtype)
        self.R2 = LayerScale(d, dtype=dtype)
        self.attn = LatentAttentionTrain(d, nh, nkv, r, use_bias, dtype = dtype)
        self.ffn = MOELayer(d, f, use_bias, num_experts, dtype = dtype)

    def __call__(self, x, labels: mx.array, m):
        x = x + self.R1(self.attn(self.n1(x), m))
        normed_x = self.n2(x)
        ffn_output = self.ffn(normed_x, labels)
        x = x + self.R2(ffn_output)
        return x
    
class LLMMini(nn.Module):
    """
    학습(Train) 전용 LLM
    """
    def __init__(self, vocab, seq, d, layers, heads, kv_heads, f=4, use_bias=True, dtype = mx.bfloat16, num_experts=1):
        super().__init__()
        self.dtype = dtype
        self.embed = Embedding(vocab, d, dtype=dtype)
        rope = rope_cache(seq, d // heads)
        self.blocks = [Block(d, heads, kv_heads, rope, f, use_bias, num_experts, dtype=dtype) for _ in range(layers)]
        self.norm = RMSNorm(d, dtype = dtype)
        self.lm_head = Linear(d, vocab, bias=False, dtype=dtype)
        self.lm_head.weight = self.embed.tok.weight

    def __call__(self, ids, labels: mx.array, m=None):
        x, _ = self.embed(ids)

        for _, b in enumerate(self.blocks):
            x = b(x, labels, m)

        x_norm = self.norm(x)
        logits = self.lm_head(x_norm)
        return logits
    
# --------------------------- Checkpoint I/O --------------------------------
def save_ckpt(model: nn.Module, optimizer: optim.Optimizer, step: int, ckpt_dir: pathlib.Path, meta: dict):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 가중치와 옵티마이저 상태를 별도의 파일에 저장
    weights_path = ckpt_dir / f"ckpt_{step:06d}_weights.safetensors"
    opt_path = ckpt_dir / f"ckpt_{step:06d}_optimizer.safetensors"
    j_path = ckpt_dir / f"ckpt_{step:06d}.json"
    
    model.save_weights(str(weights_path))

    flat_optimizer_state = dict(tree_flatten(optimizer.state))
    mx.save_safetensors(str(opt_path), flat_optimizer_state)
    
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    with open(j_path, "w") as jf:
        json.dump(meta | {"step": step, "n_params": n_params}, jf, indent=2)

    print(f"[ckpt] saved model & optimizer → {ckpt_dir}")

# [수정] load_ckpt 함수가 optimizer 객체를 받아 상태를 로드하도록 변경
def load_ckpt(model: nn.Module, optimizer: optim.Optimizer, ckpt_path: pathlib.Path):
    # ckpt_path는 이제 'ckpt_xxxxxx.json' 파일을 가리킵니다.
    if not ckpt_path.exists():
        sys.exit(f"checkpoint metadata {ckpt_path} not found")

    step = 0
    with open(ckpt_path, "r") as jf:
        meta = json.load(jf)
        step = meta.get("step", 0)

    weights_path = ckpt_path.with_name(f"ckpt_{step:06d}_weights.safetensors")
    opt_path = ckpt_path.with_name(f"ckpt_{step:06d}_optimizer.safetensors")

    if not weights_path.exists():
        sys.exit(f"Weight file {weights_path} not found")
        
    print(f"[ckpt] loading model weights ← {weights_path}...")
    model.load_weights(str(weights_path))

    if optimizer is not None and opt_path.exists():
        print(f"[ckpt] loading optimizer state ← {opt_path}...")
        opt_state = mx.load(str(opt_path))
        optimizer.state = tree_unflatten(list(opt_state.items()))
    
    print(f"[ckpt] resumed from step {step}")
    return step

# ----------------------------DATA LOADING-----------------------------------
def bin_loader(data_path: str, label_path: str, block: int, batch: int):
    """
    토큰과 라벨 파일을 함께 읽어 (x, y, labels) 튜플을 반환
    """
    ids_dtype = np.uint32
    labels_dtype = np.uint8
    
    tokens = np.memmap(data_path, dtype=ids_dtype, mode='r')
    labels = np.memmap(label_path, dtype=labels_dtype, mode='r')
    
    # 토큰과 라벨의 길이가 같은지 확인
    assert len(tokens) == len(labels), "Data and label files must have the same length"
    
    N = len(tokens) - (block + 1)
    
    while True:
        ix = np.random.randint(0, N, size=batch)
        
        # 입력 x: (batch, block)
        x_buf = np.stack([tokens[i:i+block] for i in ix])
        # 타겟 y: (batch, block)
        y_buf = np.stack([tokens[i+1:i+block+1] for i in ix])
        # 라벨 labels: (batch, block)
        l_buf = np.stack([labels[i:i+block] for i in ix])
        
        yield mx.array(x_buf), mx.array(y_buf), mx.array(l_buf)


# ------------------------------TRAIN LOOP-----------------------------------
def build_opt(lr: float, wd: float, betas = (0.9, 0.95)):
    return optim.Lion(
        learning_rate=lr,
        weight_decay=wd,
        betas=betas
    )

def lr_sched(step,total,warm,max_lr,min_lr):
    if step<warm:
        return max_lr*step/warm
    prog=(step-warm)/(total-warm)
    return min_lr + (max_lr-min_lr)*0.5*(1+math.cos(math.pi*prog))

def train(model, optimizer, steps:int, lr:float, batch:int, train_iter, val_iter,
          eval_int:int, save_int:int, block:int, ckpt_dir:pathlib.Path,
          meta:dict, start_step:int, accum:int = 1, dtype=mx.float32):

    global _DEBUG_STEP, _DEBUG_ENABLED

    cmask = causal_mask(block, dtype=dtype)
    warm = max(100, steps // 20)

    # --- 가시적인 그래디언트 Norm을 계산하는 헬퍼 함수 ---
    # def get_grad_norm(grads_tree):
    #     norm = mx.array(0.0, mx.float32)
    #     for _, g in tree_flatten(grads_tree):
    #         if g is not None:
    #             norm += mx.sum(g.astype(mx.float32) ** 2)
    #     return mx.sqrt(norm)
    
    # ------------------ 상태(State) 정의 ------------------
    # 학습과 평가에서 공통으로 사용할 상태
    # model, optimizer 등은 mutable 객체이므로 딕셔너리에 넣어도 참조가 유지됩니다.
    train_state = {
        "model": model,
        "optimizer": optimizer,
    }
    eval_state = {
        "model": model,
    }

    # [1. 컴파일된 학습 스텝]
    # 이전과 마찬가지로, Metal 버그 회피를 위해 클리핑/업데이트는 제외합니다.
    # @partial(mx.compile, inputs=train_state, outputs=train_state["model"])
    def compiled_train_loss_and_grad(x, y, labels):
        def loss_fn(mdl, x_in, y_in, labels_in):
            logits = mdl(x_in, labels_in, cmask)
            return mx.mean(nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y_in.reshape(-1), reduction="none"
            ))
        
        value_and_grad_fn = nn.value_and_grad(train_state["model"], loss_fn)
        loss, grads = value_and_grad_fn(train_state["model"], x, y, labels)
        return loss, grads

    # [2. 컴파일된 평가 스텝]
    # 평가 시에는 그래디언트가 필요 없으므로, loss 계산만 컴파일합니다.
    # @partial(mx.compile, inputs=eval_state, outputs=eval_state["model"])
    def compiled_eval_loss(x, y, labels):
        logits = model(x, labels, cmask)
        loss = mx.mean(nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none"
        ))
        return loss

    # --- 그래디언트 클리핑 함수 (컴파일되지 않음) ---
    def clip_grads(tree, max_norm=1.0):
        sq = mx.zeros((), dtype=mx.float32)
        is_finite = mx.all(mx.stack([mx.all(mx.isfinite(g)) for _, g in tree_flatten(tree)]))
        if not is_finite.item():
             print(f"Warning: NaN or Inf found in gradients at step {_DEBUG_STEP}. Skipping update.")
             return tree_map(lambda p: mx.zeros_like(p), tree)
        for _, g in tree_flatten(tree):
            sq += mx.sum(g.astype(mx.float32) ** 2)
        norm = mx.sqrt(sq)
        scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
        return tree_map(lambda g: g * scale, tree)

    # --- 메인 학습 루프 ---
    for s in range(start_step, start_step + steps):
        _DEBUG_STEP = s
        optimizer.learning_rate = lr_sched(s - start_step, steps, warm, lr, lr * 0.05)
        
        tot_loss = 0.0
        grad_acc = tree_map(lambda p: mx.zeros_like(p), model.parameters())
        # 가시적 확인을 위한 그래디언트 분석
        # current_batch_labels = None

        for _ in range(accum):
            # 컴파일된 함수 호출
            x, y, labels = next(train_iter)
            # 가시적 확인을 위한 그래디언트 분석
            # if _ == accum - 1:
            #     current_batch_labels = labels
            loss, grads = compiled_train_loss_and_grad(x, y, labels)
            grad_acc = tree_map(lambda acc, new: acc + new, grad_acc, grads)
            tot_loss += loss.item()

        grad_acc = tree_map(lambda g: g / accum, grad_acc)

        # 컴파일되지 않은 후처리 단계
        clipped_grads = clip_grads(grad_acc)
        optimizer.update(model, clipped_grads)
        mx.eval(model.parameters(), optimizer.state)

        # # 가시적 확인을 위한 그래디언트 분석
        # if (s + 1) % 10 == 0:
        #     print("-" * 50)
        #     print(f"Step {s+1} Gradient Analysis:")
            
        #     # 현재 배치에 어떤 전문가들이 있었는지 확인
        #     unique_labels_in_batch = sorted(np.unique(current_batch_labels.tolist()).tolist())
        #     print(f"  > Active experts in this batch: {unique_labels_in_batch}")

        #     # 모델의 모든 ffn 레이어(Block)를 순회하며 그래디언트 확인
        #     for i, block in enumerate(model.blocks):
        #         # block.ffn은 MOELayer 인스턴스
        #         # grad_acc 딕셔너리에서 해당 블록의 ffn 그래디언트를 가져옴
        #         ffn_grads = grad_acc['blocks'][i]['ffn']
                
        #         expert_grad_norms = []
        #         for exp_idx in range(len(block.ffn.experts)):
        #             # 각 전문가의 그래디언트만 추출
        #             expert_grad_tree = ffn_grads['experts'][exp_idx]
                    
        #             # 그래디언트 Norm 계산
        #             norm = get_grad_norm(expert_grad_tree)
        #             mx.eval(norm) # 계산 실행
        #             expert_grad_norms.append(f"{norm.item():.4f}")
                
        #         print(f"  > Block {i} FFN Grad Norms: [ {' | '.join(expert_grad_norms)} ]")
        #     print("-" * 50)

        # 로그 및 평가 (이전과 동일)
        if (s + 1) % eval_int == 0:
            vx, vy, vlabels = next(val_iter)
            # 컴파일된 평가 함수를 호출합니다.
            v_loss = compiled_eval_loss(vx, vy, vlabels)
            
            # v_loss.item()을 호출하기 전 mx.eval()을 명시적으로 호출하여 평가 그래프 계산을 완료하고 동기화
            mx.eval(v_loss)
            
            perplexity = math.exp(v_loss.item())
            print(f"{s+1:>7} | tr {tot_loss/accum:.4f} | "
                  f"Perplexity {perplexity:.2f}")
                  
        elif (s + 1) % 10 == 0 or s == start_step:
            print(f"{s+1:>7} | loss {tot_loss/accum:.4f}")

        if (s + 1) % save_int == 0:
            save_ckpt(model, optimizer, s + 1, ckpt_dir, meta)
    save_ckpt(model, optimizer, start_step + steps, ckpt_dir, meta)

# -----------------------TOKENISER / VOCAB HELPERS---------------------------
def load_encoding(name: str) -> Tuple[tiktoken.Encoding, int]:
    enc = tiktoken.get_encoding(name)
    n_vocab = enc.max_token_value + 1
    return enc, n_vocab

LLM_CONFIG: Dict[str, Dict] = {
    "custom":      dict(num_layers=16,  num_heads=16,   embd_dim=512,   block_size=1024,   use_bias=True,   LR=5e-6,  num_experts=8),
    "nano":        dict(num_layers=4,   num_heads=4,    embd_dim=512,   block_size=1024,   use_bias=True,   LR=1e-5,  num_experts=4),
    "gpt2":        dict(num_layers=12,  num_heads=12,   embd_dim=768,   block_size=1024,   use_bias=True,   LR=1e-5,  num_experts=8),
    "gpt2-medium": dict(num_layers=24,  num_heads=16,   embd_dim=1024,  block_size=1024,   use_bias=True,   LR=1e-5,  num_experts=8),
    "gpt2-large":  dict(num_layers=36,  num_heads=20,   embd_dim=1280,  block_size=1024,   use_bias=True,   LR=1e-5,  num_experts=8),
    "gpt2-xl":     dict(num_layers=48,  num_heads=25,   embd_dim=1600,  block_size=1024,   use_bias=True,   LR=1e-5,  num_experts=16),
}

# PYTHONPATH=. uv run examples/GatelessMoELLM/train.py --preset custom --train_steps 10000 --kv_heads 16
# PYTHONPATH=. uv run examples/GatelessMoELLM/train.py --preset custom --train_steps 10000 --kv_heads 16 --resume ckpt_000000.json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=LLM_CONFIG)
    ap.add_argument("--enc", default="o200k_base")
    ap.add_argument("--kv_heads", type=int, default=None)
    ap.add_argument("--data_dir", type=str, default="./")
    ap.add_argument("--train_steps", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--ckpt_dir", type=str, default="moe_ckpts")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--accum-steps", type=int, default=1, help="Number of gradient accumulation steps.")
    args = ap.parse_args()

    enc, vocab_size = load_encoding(args.enc)
    cfg = LLM_CONFIG[args.preset]
    kv_heads = args.kv_heads or max(1, cfg["num_heads"]//4)

    model = LLMMini(
        vocab_size, 
        cfg["block_size"], 
        cfg["embd_dim"], 
        cfg["num_layers"], 
        cfg["num_heads"], 
        kv_heads,
        use_bias = cfg["use_bias"],
        dtype = mx.bfloat16,
        num_experts = cfg["num_experts"]
    )
    optimizer = build_opt(cfg["LR"], 0.01)
    n_params = count_params(model.parameters()) / 1e6
    print(f"{args.preset}: {n_params:.1f}M | heads {cfg['num_heads']} / kv {kv_heads} | vocab {vocab_size}")

    start_step = 0
    if args.resume:
        start_step = load_ckpt(model, optimizer, pathlib.Path(args.ckpt_dir) / args.resume)

    if args.train_steps and args.data_dir:
        tbin_ids = pathlib.Path(args.data_dir) / "train_ids.bin"
        tbin_labels = pathlib.Path(args.data_dir) / "train_labels.bin"
        vbin_ids = pathlib.Path(args.data_dir) / "val_ids.bin"
        vbin_labels = pathlib.Path(args.data_dir) / "val_labels.bin"
        if not all([f.exists() for f in [tbin_ids, tbin_labels, vbin_ids, vbin_labels]]):
            sys.exit("One or more data/label .bin files are missing.")
        train_iter = bin_loader(str(tbin_ids), str(tbin_labels), cfg["block_size"], args.batch_size)
        val_iter = bin_loader(str(vbin_ids), str(vbin_labels), cfg["block_size"], args.batch_size)
        meta = {"preset":args.preset,"kv_heads":kv_heads,"enc":args.enc}
        train(
            model,
            optimizer,
            args.train_steps,
            cfg["LR"],
            args.batch_size,
            train_iter,
            val_iter,
            args.eval_every,
            args.save_every,
            cfg["block_size"],
            pathlib.Path(args.ckpt_dir),
            meta,start_step,
            args.accum_steps,
            dtype=mx.bfloat16)
    elif args.train_steps:
        print("--data_dir absent: using random tokens")
    else:
        print("Build OK – specify --train_steps + --data_dir to start training.")
