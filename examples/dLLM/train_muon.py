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
    rope_cache,
    create_no_effect_mask,
    count_params,
    cosine_schedule,
    q_sample_coupled
)

class ParameterContainer:
    """
    옵티마이저가 요구하는 `parameters()`와 `update()` 메서드를 제공하는
    간단한 컨테이너. nn.Module의 복잡한 동작을 피하고 안정성을 확보하기 위해 사용합니다.
    """
    def __init__(self, params_tree: dict):
        self._params = params_tree

    def parameters(self):
        return self._params

    def update(self, updates_tree: dict):
        self._params = updates_tree

class HybridMuonAdamW(optim.Optimizer):
    """
    공식 mlx.optimizers.Muon과 AdamW를 결합한 하이브리드 옵티마이저.
    """
    def __init__(self, model: nn.Module, param_groups: list, total_steps: int):
        super().__init__()
        self.total_steps = total_steps
        self.warmup_steps = max(100, total_steps // 20) if total_steps > 0 else 0
        self.current_step = 0
        
        muon_group = next((g for g in param_groups if g.get("use_muon")), None)
        adamw_group = next((g for g in param_groups if not g.get("use_muon")), None)
        
        if not muon_group or not adamw_group:
            raise ValueError("param_groups must contain one group for Muon and one for AdamW.")

        # [수정] 이제 param_groups의 params는 트리 구조를 가짐
        self.muon_params_flat = tree_flatten(muon_group["params"])
        self.adamw_params_flat = tree_flatten(adamw_group["params"])
        
        self.muon_model = ParameterContainer(muon_group["params"])
        self.adamw_model = ParameterContainer(adamw_group["params"])

        muon_group["initial_lr"] = muon_group.get("learning_rate", 3e-4)
        self.muon_optimizer = optim.Muon(
            learning_rate=muon_group["initial_lr"],
            momentum=muon_group.get("momentum", 0.95),
            weight_decay=muon_group.get("weight_decay", 0.01),
            nesterov=muon_group.get("nesterov", True),
            ns_steps=muon_group.get("ns_steps", 5)
        )

        adamw_group["initial_lr"] = adamw_group.get("learning_rate", 1e-5)
        self.adamw_optimizer = optim.AdamW(
            learning_rate=adamw_group["initial_lr"],
            betas=adamw_group.get("betas", (0.9, 0.95)),
            eps=adamw_group.get("eps", 1e-8),
            weight_decay=adamw_group.get("weight_decay", 0.01)
        )
        self.param_groups = [muon_group, adamw_group]

    def schedule_step(self):
        self.current_step += 1
        for group, optimizer in [(self.param_groups[0], self.muon_optimizer), (self.param_groups[1], self.adamw_optimizer)]:
            max_lr, min_lr = group["initial_lr"], group["initial_lr"] * 0.05
            if self.total_steps <= 0 or self.warmup_steps <= 0:
                lr = max_lr
            elif self.current_step < self.warmup_steps:
                lr = max_lr * self.current_step / self.warmup_steps
            else:
                prog = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * prog))
            optimizer.learning_rate = lr

    def apply_gradients(self, grads, params):
        # [★★핵심 수정★★]
        # optimizer.update() 대신 optimizer.apply_gradients()를 직접 호출합니다.
        
        # 1. 그래디언트를 그룹별 트리로 분리
        all_grads_dict = dict(tree_flatten(grads))
        muon_grads_tree = tree_unflatten([(k, all_grads_dict[k]) for k, _ in self.muon_params_flat])
        adamw_grads_tree = tree_unflatten([(k, all_grads_dict[k]) for k, _ in self.adamw_params_flat])
        
        # 2. 각 그룹별로 파라미터 업데이트
        # Muon 그룹
        muon_params = self.muon_model.parameters()
        updated_muon_params = self.muon_optimizer.apply_gradients(muon_grads_tree, muon_params)
        self.muon_model.update(updated_muon_params)
        
        # AdamW 그룹
        adamw_params = self.adamw_model.parameters()
        updated_adamw_params = self.adamw_optimizer.apply_gradients(adamw_grads_tree, adamw_params)
        self.adamw_model.update(updated_adamw_params)

        # 3. 업데이트된 파라미터들을 다시 하나의 트리로 병합
        final_params = {}
        final_params.update(tree_flatten(self.muon_model.parameters()))
        final_params.update(tree_flatten(self.adamw_model.parameters()))
        
        # 4. 원래 모델의 파라미터 순서대로 최종 트리를 재구성하여 반환
        final_params_flat = [(k, final_params[k]) for k, _ in tree_flatten(params)]
        return tree_unflatten(final_params_flat)

    @property
    def state(self):
        return {"muon": self.muon_optimizer.state, "adamw": self.adamw_optimizer.state}

    @state.setter
    def state(self, new_state):
        if "muon" in new_state: self.muon_optimizer.state = new_state["muon"]
        if "adamw" in new_state: self.adamw_optimizer.state = new_state["adamw"]


def create_muon_param_groups(
    model: nn.Module,
    muon_config: dict,
    adamw_config: dict,
    exclude_names: list = ["norm", "bias"]
):
    """
    [수정] 키 구조를 유지하는 부분 파라미터 트리를 반환하도록 수정
    """
    muon_params_flat, adamw_params_flat = [], []
    for name, param in tree_flatten(model.parameters()):
        is_excluded = any(ex_name in name for ex_name in exclude_names) or param.ndim < 2
        (adamw_params_flat if is_excluded else muon_params_flat).append((name, param))

    print(f"--- Optimizer Grouping ---")
    print(f"Muon group size: {len(muon_params_flat)} parameter tensors")
    print(f"AdamW group size: {len(adamw_params_flat)} parameter tensors")

    return [
        {"params": tree_unflatten(muon_params_flat), "use_muon": True, **muon_config},
        {"params": tree_unflatten(adamw_params_flat), "use_muon": False, **adamw_config}
    ]


class Block(nn.Module):
    def __init__(self, d, nh, nkv, r, f, use_bias, dtype):
        super().__init__()
        self.n1 = RMSNorm(d, dtype=dtype)
        self.R1 = LayerScale(d, dtype=dtype)
        self.n2 = RMSNorm(d, dtype=dtype)
        self.R2 = LayerScale(d, dtype=dtype)
        # Latent Attention 사용
        self.attn = LatentAttentionTrain(d, nh, nkv, r, use_bias, dtype=dtype)
        # FFN으로 GRN 사용
        self.ffn = GRN(d, f, use_bias, dtype = dtype)

    def __call__(self, x, m):
        x = x + self.R1(self.attn(self.n1(x), m))
        normed_x = self.n2(x)
        ffn_output = self.ffn(normed_x)
        x = x + self.R2(ffn_output)
        return x # 튜플 반환
    
class dLLMMini(nn.Module):
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

        for b in self.blocks:
            x = b(x, m)
            
        x_norm = self.norm(x)
        logits = self.lm_head(x_norm)
        return logits
    
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

def bin_loader(data_path: str, block: int, batch: int):
    """
    토큰과 라벨 파일을 함께 읽어 (x, y, labels) 튜플을 반환
    """
    ids_dtype = np.uint32
    labels_dtype = np.uint8
    
    tokens = np.memmap(data_path, dtype=ids_dtype, mode='r')
    
    N = len(tokens) - (block + 1)
    
    while True:
        ix = np.random.randint(0, N, size=batch)
        
        # 입력 x: (batch, block)
        x_buf = np.stack([tokens[i:i+block] for i in ix])
        # 타겟 y: (batch, block)
        y_buf = np.stack([tokens[i+1:i+block+1] for i in ix])
        
        yield mx.array(x_buf), mx.array(y_buf)

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

def train(model, optimizer, steps:int, max_lr: float, train_iter, val_iter,
          eval_int:int, save_int:int, block:int, ckpt_dir:pathlib.Path,
          meta:dict, start_step:int, accum:int = 1, dtype=mx.float32, mask_token_id: int = 200019):
    
    diffusion_timesteps = 1000
    alphas_cumprod = cosine_schedule(diffusion_timesteps, dtype=dtype)

    cmask = create_no_effect_mask(block, dtype)
    warm = max(100, steps // 20)
    min_lr = max_lr * 0.05

    train_state = {
        "model": model,
        "optimizer": optimizer,
    }

    # 1. 컴파일된 학습 스텝 - 클리핑/업데이트는 제외, 메모리가 증가하고 오히려 연산 능력이 떨어짐.
    # @partial(mx.compile, inputs=train_state, outputs=train_state["model"])
    def compiled_train_loss_and_grad(x):
        # 1. 배치마다 랜덤 타임스텝 샘플링
        t = mx.random.randint(0, diffusion_timesteps, (x.shape[0],))

        (x_t1, noise_mask1), (x_t2, noise_mask2) = q_sample_coupled(
            x, t, alphas_cumprod, mask_token_id
        )

        def loss_fn(model):
            # 2. 첫 번째 샘플에 대한 예측 및 손실 계산
            pred_x_start_logits1 = model(x_t1, cmask)
            loss1_per_token = nn.losses.cross_entropy(pred_x_start_logits1, x, reduction="none")
            masked_loss1 = mx.sum(loss1_per_token * noise_mask1) / (mx.sum(noise_mask1) + 1e-8)

            # 3. 두 번째 샘플에 대한 예측 및 손실 계산
            pred_x_start_logits2 = model(x_t2, cmask)
            loss2_per_token = nn.losses.cross_entropy(pred_x_start_logits2, x, reduction="none")
            masked_loss2 = mx.sum(loss2_per_token * noise_mask2) / (mx.sum(noise_mask2) + 1e-8)
            
            # 4. 두 손실을 평균내어 최종 손실 반환
            total_loss = (masked_loss1 + masked_loss2) / 2.0
            return total_loss
        
        value_and_grad_fn = nn.value_and_grad(train_state["model"], loss_fn)
        loss, grads = value_and_grad_fn(train_state["model"])

        return loss, grads

    # 2. 컴파일된 평가 스텝 - 평가 시에는 그래디언트가 필요 없으므로, loss 계산만
    # @partial(mx.compile, inputs=eval_state, outputs=eval_state["model"])
    def compiled_eval_loss(x):
        # 1. 배치마다 랜덤 타임스텝 샘플링
        t = mx.random.randint(0, diffusion_timesteps, (x.shape[0],))

        (x_t1, noise_mask1), (x_t2, noise_mask2) = q_sample_coupled(
            x, t, alphas_cumprod, mask_token_id
        )

        # 2. 첫 번째 샘플에 대한 예측 및 손실 계산
        pred_x_start_logits1 = model(x_t1, cmask)
        loss1_per_token = nn.losses.cross_entropy(pred_x_start_logits1, x, reduction="none")
        masked_loss1 = mx.sum(loss1_per_token * noise_mask1) / (mx.sum(noise_mask1) + 1e-8)

        # 3. 두 번째 샘플에 대한 예측 및 손실 계산
        pred_x_start_logits2 = model(x_t2, cmask)
        loss2_per_token = nn.losses.cross_entropy(pred_x_start_logits2, x, reduction="none")
        masked_loss2 = mx.sum(loss2_per_token * noise_mask2) / (mx.sum(noise_mask2) + 1e-8)
        
        # 4. 두 손실을 평균내어 최종 손실 반환
        total_loss = (masked_loss1 + masked_loss2) / 2.0
        return total_loss

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
        if isinstance(optimizer, (HybridMuonAdamW)): # AdamW도 스케줄링이 필요하다면 여기에 추가
            optimizer.schedule_step()
        else:
            # Lion 및 다른 표준 옵티마이저를 위한 외부 스케줄러
            # 현재 스텝 (s)를 사용 (start_step부터 시작하므로)
            current_step = s + 1
            if current_step < warm:
                lr = max_lr * current_step / warm
            else:
                prog = (current_step - warm) / (steps - warm)
                lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * prog))
            # [핵심 수정] 계산된 학습률을 옵티마이저에 적용
            optimizer.learning_rate = lr
        
        tot_loss = 0.0
        grad_acc = tree_map(lambda p: mx.zeros_like(p), model.parameters())

        for _ in range(accum):
            # 컴파일된 함수 호출
            x, _ = next(train_iter)
            loss, grads = compiled_train_loss_and_grad(x)
            grad_acc = tree_map(lambda acc, new: acc + new, grad_acc, grads)
            tot_loss += loss.item()

        grad_acc = tree_map(lambda g: g / accum, grad_acc)

        # 컴파일되지 않은 후처리 단계
        clipped_grads = clip_grads(grad_acc)
        optimizer.update(model, clipped_grads)
        mx.eval(model.parameters(), optimizer.state)

        # 로그 및 평가 (이전과 동일)
        if (s + 1) % eval_int == 0:
            vx, _ = next(val_iter)
            
            # 컴파일된 평가 함수를 호출합니다.
            v_loss = compiled_eval_loss(vx)
            
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
    """
    tiktoken 인코더를 로드하고, <|mask|> 특수 토큰을 추가합니다.
    반환값: (인코더, 어휘집 크기, 마스크 토큰 ID)
    """
    # 기본 인코더 로드
    enc = tiktoken.get_encoding(name)
    
    # 새로운 특수 토큰 정의. 기존 어휘집 크기를 ID로 사용합니다.
    mask_token = "<|mask|>"
    mask_token_id = enc.max_token_value + 1
    
    # tiktoken 인코더에 특수 토큰 등록
    special_tokens = {mask_token: mask_token_id}
    
    # tiktoken v0.7.0 이상에서는 아래와 같이 확장된 인코더를 만듭니다.
    enc = tiktoken.Encoding(
        name=f"{name}_with_mask",
        pat_str=enc._pat_str,
        mergeable_ranks=enc._mergeable_ranks,
        special_tokens={**enc._special_tokens, **special_tokens}
    )

    # 새로운 어휘집 크기
    n_vocab = enc.n_vocab # enc.max_token_value + 1 대신 n_vocab 사용
    
    print(f"Tokenizer '{name}' loaded. Vocabulary size: {enc.max_token_value + 1}")
    print(f"Added special token '{mask_token}' with ID: {mask_token_id}")
    print(f"New vocabulary size: {n_vocab}")
    
    return enc, n_vocab, mask_token_id

LLM_CONFIG: Dict[str, Dict] = {
    "custom":      dict(num_layers=12,  num_heads=16,    embd_dim=512,   block_size=1024,   use_bias=True,   LR=1e-5),
    "nano":        dict(num_layers=4,   num_heads=4,    embd_dim=512,   block_size=1024,   use_bias=True,   LR=1e-5),
    "gpt2":        dict(num_layers=12,  num_heads=12,   embd_dim=768,   block_size=1024,   use_bias=True,   LR=1e-5),
    "gpt2-medium": dict(num_layers=24,  num_heads=16,   embd_dim=1024,  block_size=1024,   use_bias=True,   LR=1e-5),
    "gpt2-large":  dict(num_layers=36,  num_heads=20,   embd_dim=1280,  block_size=1024,   use_bias=True,   LR=1e-5),
    "gpt2-xl":     dict(num_layers=48,  num_heads=25,   embd_dim=1600,  block_size=1024,   use_bias=True,   LR=1e-5),
}

# PYTHONPATH=. uv run examples/dLLM/train_muon.py --preset custom --kv_heads 16 --train_steps 3000 --optimizer muon --ckpt_dir dllmmuon_ckptss
# PYTHONPATH=. uv run examples/dLLM/train_muon.py --preset custom --kv_heads 16 --train_steps 2000 --optimizer muon --ckpt_dir dllmmuon_ckptss --resume ckpt_001000.json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=LLM_CONFIG)
    ap.add_argument("--optimizer", type=str, default="lion", choices=["lion", "muon"], help="Optimizer to use.")
    ap.add_argument("--enc", default="o200k_base")
    ap.add_argument("--kv_heads", type=int, default=None)
    ap.add_argument("--data_dir", type=str, default="./")
    ap.add_argument("--train_steps", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--ckpt_dir", type=str, default="dllmmuon_ckpts")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--accum-steps", type=int, default=1, help="Number of gradient accumulation steps.")
    args = ap.parse_args()

    enc, vocab_size, mask_token_id = load_encoding(args.enc)
    cfg = LLM_CONFIG[args.preset]
    kv_heads = args.kv_heads or max(1, cfg["num_heads"]//4)

    model = dLLMMini(
        vocab_size, 
        cfg["block_size"], 
        cfg["embd_dim"], 
        cfg["num_layers"], 
        cfg["num_heads"], 
        kv_heads, 
        use_bias=cfg["use_bias"],
        dtype = mx.bfloat16,
    )

    if args.optimizer == 'muon':
        print("Using HybridMuonAdamW optimizer.")
        # Muon은 AdamW보다 높은 학습률을 사용하는 경향이 있음
        muon_lr = 3e-4
        adamw_lr = cfg['LR'] # 기존 LR은 AdamW 부분에 사용
        
        param_groups = create_muon_param_groups(
            model=model,
            muon_config={"learning_rate": muon_lr, "weight_decay": 0.01},
            adamw_config={"learning_rate": adamw_lr, "weight_decay": 0.01}
        )
        optimizer = HybridMuonAdamW(model, param_groups, total_steps=args.train_steps)
    else:
        print("Using Lion optimizer.")
        optimizer = optim.Lion(learning_rate=cfg["LR"], weight_decay=0.01)

    n_params = count_params(model.parameters()) / 1e6
    print(f"{args.preset}: {n_params:.1f}M | heads {cfg['num_heads']} / kv {kv_heads} | vocab {vocab_size}")

    start_step = 0
    if args.resume:
        start_step = load_ckpt(model, optimizer, pathlib.Path(args.ckpt_dir) / args.resume)

    if args.train_steps and args.data_dir:
        tbin_ids = pathlib.Path(args.data_dir) / "train_ids.bin"
        vbin_ids = pathlib.Path(args.data_dir) / "val_ids.bin"
        if not all([f.exists() for f in [tbin_ids, vbin_ids]]):
            sys.exit("One or more data/label .bin files are missing.")
        train_iter = bin_loader(str(tbin_ids), cfg["block_size"], args.batch_size)
        val_iter = bin_loader(str(vbin_ids), cfg["block_size"], args.batch_size)
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
            dtype=mx.bfloat16,
            mask_token_id=mask_token_id)
    elif args.train_steps:
        print("--data_dir absent: using random tokens")
    else:
        print("Build OK – specify --train_steps + --data_dir to start training.")
