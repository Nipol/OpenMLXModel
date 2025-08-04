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
import math, argparse, os, sys, time, pathlib, random, json

from typing import Dict, Tuple, Optional
from functools import partial
from mlx.utils import tree_unflatten, tree_flatten, tree_map
from attentions.LatentAttention import LatentAttentionTrain
from ffns.GRNFFN import GRN
from embedding import Embedding
from common import (
    Linear,
    RMSNorm,
    LayerScale,
    rope_cache,
    causal_mask,
    count_params
)

class Block(nn.Module):
    def __init__(self, d, nh, nkv, r, f, use_bias, dtype):
        super().__init__()
        self.n1 = RMSNorm(d, dtype=dtype)
        self.R1 = LayerScale(d, dtype=dtype)
        self.n2 = RMSNorm(d, dtype=dtype)
        self.R2 = LayerScale(d, dtype=dtype)
        self.attn = LatentAttentionTrain(d, nh, nkv, r, use_bias, dtype = dtype)
        self.ffn = GRN(d, f, use_bias, dtype = dtype)

    def __call__(self, x, m):
        x = x + self.R1(self.attn(self.n1(x), m))
        normed_x = self.n2(x)
        ffn_output, aux_loss = self.ffn(normed_x)
        x = x + self.R2(ffn_output)
        return x, aux_loss # 튜플 반환
    
class LLMMini(nn.Module):
    """
    학습(Train) 전용 LLM
    """
    def __init__(self, vocab, seq, d, layers, heads, kv_heads, f=4, use_bias=True, dtype = mx.float32):
        super().__init__()
        self.embed=Embedding(vocab, d, dtype=dtype)
        rope = rope_cache(seq, d // heads)
        self.blocks=[Block(d, heads, kv_heads, rope, f, use_bias, dtype=dtype) for _ in range(layers)]
        self.norm=RMSNorm(d, dtype = dtype)
        self.lm_head=Linear(d, vocab, bias=False, dtype=dtype)
        self.lm_head.weight=self.embed.tok.weight

    def __call__(self, ids, m=None):
        x, _ = self.embed(ids)
        total_aux_loss = mx.array(0.0, dtype=x.dtype)

        for _, b in enumerate(self.blocks):
            x, aux_loss = b(x, m)
            total_aux_loss += aux_loss
            
        x_norm = self.norm(x)
        logits = self.lm_head(x_norm)
        return logits, total_aux_loss
    
# --------------------------- Checkpoint I/O --------------------------------
def save_ckpt(model: nn.Module, optimizer: optim.Optimizer, step: int, ckpt_dir: pathlib.Path, meta: dict):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 가중치와 옵티마이저 상태를 별도의 파일에 저장
    weights_path = ckpt_dir / f"ckpt_{step:06d}_weights.safetensors"
    opt_path = ckpt_dir / f"ckpt_{step:06d}_optimizer.npz"
    j_path = ckpt_dir / f"ckpt_{step:06d}.json"
    
    model.save_weights(str(weights_path))
    
    # 옵티마이저 상태를 .npz 파일로 저장
    mx.savez(opt_path, **dict(tree_flatten(optimizer.state)))
    
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
    opt_path = ckpt_path.with_name(f"ckpt_{step:06d}_optimizer.npz")

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
def openweb_bin_loader(path:str, block:int, batch:int):
    """Yields MX arrays (x,y) shaped (batch, block) from mem‑mapped tokens."""
    dtype = np.uint32 
    tokens = np.memmap(path, dtype=dtype, mode='r')
    N = tokens.shape[0] - (block+1)
    while True:
        ix = np.random.randint(0, N, size=batch)
        buf = np.stack([tokens[i:i+block+1] for i in ix])
        yield mx.array(buf[:,:block]), mx.array(buf[:,1:])

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

def train(model, optimizer, steps:int, lr:float, train_iter, val_iter,
          eval_int:int, save_int:int, block:int, ckpt_dir:pathlib.Path,
          meta:dict, start_step:int, accum:int = 1, dtype=mx.float32):

    cmask = causal_mask(block, dtype=dtype)
    warm = max(100, steps // 20)

    train_state = {
        "model": model,
        "optimizer": optimizer,
    }
    eval_state = {
        "model": model,
    }

    # 1. 컴파일된 학습 스텝 - 클리핑/업데이트는 제외합니다.
    @partial(mx.compile, inputs=train_state, outputs=train_state["model"])
    def compiled_train_loss_and_grad(x, y):
        def loss_fn(mdl, x_in, y_in):
            logits, _ = mdl(x_in, cmask)
            return mx.mean(nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y_in.reshape(-1), reduction="none"
            ))
        
        value_and_grad_fn = nn.value_and_grad(train_state["model"], loss_fn)
        loss, grads = value_and_grad_fn(train_state["model"], x, y)
        return loss, grads

    # 2. 컴파일된 평가 스텝 - 평가 시에는 그래디언트가 필요 없으므로, loss 계산만
    @partial(mx.compile, inputs=eval_state, outputs=eval_state["model"])
    def compiled_eval_loss(x, y):
        logits, _ = model(x, cmask)
        loss = mx.mean(nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none"
        ))
        return loss

    # --- 그래디언트 클리핑 함수 (컴파일되지 않음) ---
    def clip_grads(tree, max_norm=1.0):
        sq = mx.zeros((), dtype=mx.float32)
        for _, g in tree_flatten(tree):
            sq += mx.sum(g.astype(mx.float32) ** 2)
        norm = mx.sqrt(sq)
        scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
        return tree_map(lambda g: g * scale, tree)

    # --- 메인 학습 루프 ---
    for s in range(start_step, start_step + steps):
        optimizer.learning_rate = lr_sched(s - start_step, steps, warm, lr, lr * 0.05)
        
        tot_loss = 0.0
        grad_acc = tree_map(lambda p: mx.zeros_like(p), model.parameters())

        for _ in range(accum):
            # 컴파일된 함수 호출
            x, y = next(train_iter)
            loss, grads = compiled_train_loss_and_grad(x, y)
            grad_acc = tree_map(lambda acc, new: acc + new, grad_acc, grads)
            tot_loss += loss.item()

        grad_acc = tree_map(lambda g: g / accum, grad_acc)

        # 컴파일되지 않은 후처리 단계
        clipped_grads = clip_grads(grad_acc)
        optimizer.update(model, clipped_grads)
        mx.eval(model.parameters(), optimizer.state)

        # 로그 및 평가 (이전과 동일)
        if (s + 1) % eval_int == 0:
            vx, vy = next(val_iter)
            
            # 컴파일된 평가 함수를 호출합니다.
            v_loss = compiled_eval_loss(vx, vy)
            
            # v_loss.item()을 호출하기 전에 mx.eval()을 명시적으로 호출하여 평가 그래프 계산을 완료하고 동기화합니다.
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
    "custom":      dict(num_layers=12,  num_heads=8,    embd_dim=512,   block_size=1024,   use_bias=True,   LR=5e-6),
    "nano":        dict(num_layers=4,   num_heads=4,    embd_dim=512,   block_size=1024,   use_bias=True,   LR=1e-5),
    "gpt2":        dict(num_layers=12,  num_heads=12,   embd_dim=768,   block_size=1024,   use_bias=True,   LR=1e-5),
    "gpt2-medium": dict(num_layers=24,  num_heads=16,   embd_dim=1024,  block_size=1024,   use_bias=True,   LR=1e-5),
    "gpt2-large":  dict(num_layers=36,  num_heads=20,   embd_dim=1280,  block_size=1024,   use_bias=True,   LR=1e-5),
    "gpt2-xl":     dict(num_layers=48,  num_heads=25,   embd_dim=1600,  block_size=1024,   use_bias=True,   LR=1e-5),
}

# PYTHONPATH=. uv run examples/LLM/train.py --preset custom --train_steps 1000

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
    ap.add_argument("--ckpt_dir", type=str, default="llm_ckpts")
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
        use_bias=cfg["use_bias"],
        dtype = mx.bfloat16,
    )
    optimizer = build_opt(cfg["LR"], 0.01)
    n_params = count_params(model.parameters()) / 1e6
    print(f"{args.preset}: {n_params:.1f}M | heads {cfg['num_heads']} / kv {kv_heads} | vocab {vocab_size}")

    start_step = 0
    if args.resume:
        start_step = load_ckpt(model, optimizer, pathlib.Path(args.ckpt_dir) / args.resume)

    if args.train_steps and args.data_dir:
        tbin = pathlib.Path(args.data_dir)/"train.bin"; vbin = pathlib.Path(args.data_dir)/"val.bin"
        if not (tbin.exists() and vbin.exists()):
            sys.exit("train.bin/val.bin missing")
        train_iter = openweb_bin_loader(str(tbin), cfg["block_size"], args.batch_size)
        val_iter   = openweb_bin_loader(str(vbin), cfg["block_size"], args.batch_size)
        meta = {"preset":args.preset,"kv_heads":kv_heads,"enc":args.enc}
        train(
            model,
            optimizer,
            args.train_steps,
            cfg["LR"],
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
