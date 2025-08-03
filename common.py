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

# ---------------------------------------------------------------------------
#                               COMMON
# ---------------------------------------------------------------------------
import math
import mlx.core as mx
import mlx.nn as nn
from functools import partial

def count_params(tree):
    if isinstance(tree, mx.array):          # 실제 텐서
        return tree.size
    if isinstance(tree, dict):             # 서브모듈 dict
        return sum(count_params(v) for v in tree.values())
    if isinstance(tree, (list, tuple)):    # 리스트 형태
        return sum(count_params(v) for v in tree)
    return 0

def cosine_schedule(timesteps: int, s: float = 0.008, dtype=mx.float32) -> mx.array:
    """ 코사인 노이즈 스케줄에 따라 alphas_cumprod를 생성합니다. """
    steps = timesteps + 1
    x = mx.linspace(0, timesteps, steps)
    alphas_cumprod = mx.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0] # 정규화
    return alphas_cumprod.astype(dtype)

def create_no_effect_mask(sequence_length: int, dtype=mx.float32) -> mx.array:
    """
    모든 요소가 0으로 채워져 어텐션에 아무런 영향을 주지 않는 마스크를 생성합니다.
    dLLM의 양방향 어텐션을 구현할 때 사용됩니다.

    Args:
        sequence_length (int): 시퀀스의 길이.
        dtype (mx.Dtype, optional): 데이터 타입. Defaults to mx.float32.

    Returns:
        mx.array: (sequence_length, sequence_length) 크기의 0으로 채워진 행렬.
    """
    # 어텐션 마스크는 보통 (seq_len, seq_len) 또는 (batch, heads, seq_len, seq_len) 형태입니다.
    # 어텐션 구현에 따라 브로드캐스팅이 가능하도록 (1, 1, seq_len, seq_len) 등으로 만들 수도 있습니다.
    # 가장 일반적인 형태는 (seq_len, seq_len) 입니다.
    return mx.zeros((sequence_length, sequence_length), dtype=dtype)

def causal_mask(T: int, dtype=mx.float32):
    m = mx.ones((T,T),dtype=mx.bool_)
    return mx.where(mx.tril(m),0.,-mx.inf)[None,None].astype(dtype)

def sliding_window_causal_mask(seq_len: int, window_size: int, dtype=mx.float32):
    """
    Sliding Window Causal Attention을 위한 마스크를 생성합니다.
    토큰은 자기 자신과 window_size - 1 만큼의 이전 토큰만 볼 수 있습니다.

    Args:
        seq_len (int): 시퀀스의 전체 길이.
        window_size (int): 어텐션 윈도우의 크기.

    Returns:
        mx.array: (1, 1, seq_len, seq_len) 형태의 어텐션 마스크.
    """
    # 1. 쿼리(행)와 키(열)의 인덱스를 나타내는 격자(grid)를 생성합니다.
    q_indices = mx.arange(seq_len)[:, None]
    k_indices = mx.arange(seq_len)[None, :]

    # 2. 두 가지 조건을 정의합니다.
    #    a) 인과성(Causality) 조건: 키는 쿼리보다 앞에 있거나 같아야 함.
    causal_cond = k_indices <= q_indices
    #    b) 윈도우(Window) 조건: 키와 쿼리의 거리 차이가 window_size보다 작아야 함.
    window_cond = (q_indices - k_indices) < window_size

    # 3. 두 조건을 모두 만족하는 위치를 찾습니다.
    combined_mask = mx.logical_and(causal_cond, window_cond)

    # 4. 조건을 만족하는 곳은 0.0, 아닌 곳은 -mx.inf로 채운 마스크를 생성합니다.
    mask = mx.where(combined_mask, 0.0, -mx.inf)

    # 5. 배치와 헤드 차원을 추가하여 최종 마스크를 반환합니다.
    return mask[None, None].astype(dtype)

def bidirectional_sliding_window_mask(seq_len: int, window_size: int, dtype=mx.float32):
    """
    Diffusion LLM을 위한 양방향 슬라이딩 윈도우 마스크를 생성합니다.
    
    Args:
        seq_len (int): 전체 시퀀스 길이.
        window_size (int): 각 토큰이 주목할 양쪽 주변 토큰의 수 (반쪽 길이).
                           (예: window_size=256이면, 총 512+1개의 토큰을 봄)

    Returns:
        mx.array: 양방향 슬라이딩 윈도우 마스크.
    """
    # 각 토큰의 인덱스를 나타내는 행렬 생성
    q_indices = mx.arange(seq_len)[:, None]
    k_indices = mx.arange(seq_len)[None, :]
    
    # 윈도우 바깥쪽에 있는지 확인
    # |q_idx - k_idx| > window_size 이면 마스킹
    distance = mx.abs(q_indices - k_indices)
    is_outside_window = (distance > window_size)
    
    # 윈도우 바깥쪽은 -inf, 안쪽은 0.0으로 설정
    mask = mx.where(is_outside_window, -mx.inf, 0.0)
    
    return mask[None, None, :, :].astype(dtype)

@partial(mx.compile)
def q_sample(x_start: mx.array, t: mx.array, alphas_cumprod: mx.array, mask_token_id: int):
    """
    Forward-diffusion: 원본 x_start에 t 스텝만큼 노이즈를 추가합니다.
    여기서는 'absorb' 방식의 노이즈, 즉 토큰을 [MASK]로 대체합니다.
    """
    # 1. 현재 타임스텝 t에 해당하는 누적 알파 값(보존 비율)을 가져옵니다.
    alpha_t = alphas_cumprod[t]  # shape: (batch_size,)
    
    # 2. 각 시퀀스에서 보존할 토큰의 수를 계산합니다.
    num_tokens_to_keep = (alpha_t * x_start.shape[1]).astype(mx.int32)
    
    # 3. 마스킹할 토큰을 랜덤하게 선택하기 위한 노이즈를 생성합니다.
    noise = mx.random.uniform(shape=x_start.shape)
    
    # 4. `argsort` 트릭으로 각 토큰의 순위를 매깁니다.
    #    순위가 `num_tokens_to_keep`보다 낮은 토큰들이 보존됩니다.
    ranks = mx.argsort(mx.argsort(noise, axis=1), axis=1)
    keep_mask = ranks < num_tokens_to_keep[:, None]
    
    # 5. `[MASK]` 토큰으로 채워진 텐서와 `where`를 이용해 노이즈를 주입합니다.
    masked_x = mx.full(x_start.shape, mask_token_id, dtype=x_start.dtype)
    x_t = mx.where(keep_mask, x_start, masked_x)
    
    # 6. 손실 계산을 위해 어떤 토큰이 마스킹되었는지 알려주는 마스크를 반환합니다.
    #    (True = 마스킹됨, False = 원본 유지)
    noise_mask = ~keep_mask
    
    return x_t, noise_mask

@partial(mx.compile)
def q_sample_coupled(x_start: mx.array, t: mx.array, alphas_cumprod: mx.array, mask_token_id: int):
    """
    Forward-diffusion with Coupled-Sampling based on DiffuCoder.
    
    원본 x_start에 t 스텝만큼 노이즈를 추가하되, 두 개의 상호보완적인
    (complementary) 마스크를 생성하여 분산을 줄이고 학습 효율을 높입니다.

    Args:
        x_start (mx.array): 원본 토큰 시퀀스 (batch_size, seq_len)
        t (mx.array): 각 시퀀스에 적용할 타임스텝 (batch_size,)
        alphas_cumprod (mx.array): 타임스텝별 누적 알파 값(보존 비율)
        mask_token_id (int): [MASK] 토큰의 ID

    Returns:
        tuple: 두 개의 상호보완적인 학습 샘플 쌍.
               ((x_t1, noise_mask1), (x_t2, noise_mask2))
               각각 (마스킹된 시퀀스, 손실 계산용 노이즈 마스크)를 포함합니다.
    """
    # 1. 현재 타임스텝 t에 해당하는 누적 알파 값(보존 비율)을 가져옵니다.
    alpha_t = alphas_cumprod[t]  # shape: (batch_size,)
    
    # 2. 각 시퀀스에서 보존할 토큰의 수를 계산합니다.
    #    이것이 첫 번째 샘플(sample 1)의 보존 개수가 됩니다.
    num_tokens_to_keep_1 = (alpha_t * x_start.shape[1]).astype(mx.int32)

    # 3. 마스킹할 토큰을 랜덤하게 선택하기 위한 노이즈를 생성하고 순위를 매깁니다.
    #    argsort 트릭은 각 토큰에 0부터 (seq_len-1)까지의 고유한 순위를 부여합니다.
    noise = mx.random.uniform(shape=x_start.shape)
    ranks = mx.argsort(mx.argsort(noise, axis=1), axis=1)

    # 4. 첫 번째 마스크(mask 1) 생성
    #    순위가 낮은 토큰들을 보존합니다.
    keep_mask_1 = ranks < num_tokens_to_keep_1[:, None]
    
    # 5. 두 번째 마스크(mask 2) 생성: 첫 번째 마스크의 '보완'
    #    첫 번째에서 보존된 토큰은 마스킹하고, 마스킹된 토큰은 보존합니다.
    keep_mask_2 = ~keep_mask_1

    # 6. 마스킹된 텐서 준비
    masked_x_tensor = mx.full(x_start.shape, mask_token_id, dtype=x_start.dtype)

    # 7. 두 개의 상호보완적인 샘플 생성
    x_t1 = mx.where(keep_mask_1, x_start, masked_x_tensor)
    x_t2 = mx.where(keep_mask_2, x_start, masked_x_tensor)

    # 8. 각 샘플에 대한 손실 계산용 노이즈 마스크 생성
    #    True = 마스킹됨 (이 위치에서 손실을 계산해야 함)
    noise_mask1 = ~keep_mask_1
    noise_mask2 = ~keep_mask_2

    return (x_t1, noise_mask1), (x_t2, noise_mask2)

class Linear(nn.Module):
    """
    Mixed Precision을 지원하는 Linear 레이어.
    - 가중치(weight)는 (out_f, in_f) shape의 Uniform 분포로 초기화됨.
    - bias가 활성화된 경우, (out_f,) shape의 Uniform 분포에서 초기화된 값을 사용.
    - bias가 비활성화된 경우, 0으로 채워진 벡터를 출력에 더함.
    
    Args:
        in_f (int): 입력 feature 수.
        out_f (int): 출력 feature 수.
        bias (bool): bias 항 추가 여부.
        dtype: 가중치 및 bias의 데이터 타입.
    """
    def __init__(self, in_f:int, out_f:int, bias:bool=True, dtype=mx.float32):
        super().__init__()
        lim = 1.0 / math.sqrt(in_f) / 2.0
        self.w = mx.random.uniform(low=-lim, high=lim, shape=(out_f, in_f), dtype=dtype)
        self.b = mx.random.uniform(low=-lim, high=lim, shape=(out_f,), dtype=dtype) if bias else mx.zeros((out_f,), dtype=dtype)

    def __call__(self, x:mx.array):
        return mx.matmul(x, self.w.T) + self.b
    
    @staticmethod
    def functional_forward(x: mx.array, params: dict) -> mx.array:
        """
        vmap과 같은 함수 변환을 위한 '함수형 경로'.
        params 딕셔너리에서 가중치와 편향을 명시적으로 받아 사용.
        """
        # params 딕셔너리는 __init__에서 'bias_vector'를 항상 생성하므로,
        # .get() 없이 직접 접근 가능
        return mx.matmul(x, params['w'].T) + params['b']

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    입력 텐서를 마지막 차원 기준으로 평균 제곱근(RMS, Root Mean Square)으로 정규화한다.
    입력 x에 대해, RMSNorm은 아래 수식으로 계산된다.

        y = w * (x / sqrt(mean(x ** 2, axis=-1, keepdims=True) + eps))

    여기서 w는 학습 가능한 스케일 파라미터(벡터), eps는 수치적 안정성을 위한 작은 상수이다.

    Args:
        d (int): 정규화될 마지막 차원의 크기(=스케일 파라미터 w의 길이).
        EPS (float, optional): 수치적 안정성을 위한 작은 상수(epsilon).
        dtype: 파라미터의 데이터 타입.

    입력:
        x (mx.array): shape (..., d). 마지막 차원이 d이어야 한다.

    반환:
        mx.array: 입력과 동일한 shape. 마지막 차원은 정규화 및 스케일이 적용됨.
    """
    def __init__(self, d:int, EPS=1e-4, dtype=mx.float32):
        super().__init__()
        self.eps = EPS
        self.w = mx.ones((d, ), dtype=dtype)

    def __call__(self, x):
        return (x / mx.sqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)) * self.w

class LayerScale(nn.Module):
    def __init__(self, d, init=0.2, dtype=mx.float32):
        super().__init__()
        self.s = mx.full((d, ), init, dtype=dtype)

    def __call__(self,x):
        return x*self.s

def rope_cache(seq:int, d:int, base:int=10_000, dtype=mx.float32):
    half=d//2
    inv=1.0/(base**(mx.arange(half)/half))
    t=mx.arange(seq)[:,None]
    freqs=t*inv[None]
    return mx.concat([mx.cos(freqs),mx.sin(freqs)],-1).astype(dtype)

@mx.compile
def apply_rope(x: mx.array, freqs: mx.array):
    """
    주어진 주파수(freqs)를 사용하여 입력 텐서(x)에 RoPE를 적용합니다.
    이 함수는 입력 x와 freqs의 마지막 차원 크기가 동일하다고 가정합니다.

    Args:
        x (mx.array): RoPE를 적용할 텐서. Shape: (..., seq_len, rope_dim)
        freqs (mx.array): RoPE 주파수 텐서. Shape: (seq_len, rope_dim)

    Returns:
        mx.array: RoPE가 적용된 텐서.
    """
    # 입력 텐서와 주파수 텐서를 실수부/허수부로 분리
    x_r, x_i = mx.split(x, 2, axis=-1)
    freqs_cos, freqs_sin = mx.split(freqs, 2, axis=-1)

    # 브로드캐스팅을 위해 freqs의 차원을 확장 (예: (S, D/2) -> (1, 1, S, D/2))
    # MLX의 자동 브로드캐스팅이 대부분의 경우 처리해 줍니다.
    
    # 복소수 곱셈을 사용한 회전 변환: (x_r + i*x_i) * (cos + i*sin)
    rotated_r = x_r * freqs_cos - x_i * freqs_sin
    rotated_i = x_r * freqs_sin + x_i * freqs_cos

    return mx.concatenate([rotated_r, rotated_i], axis=-1)

def one_hot(indices: mx.array, num_classes: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    """
    MLX에서 one-hot 인코딩을 수행하는 함수.

    Args:
        indices (mx.array): 정수형 라벨(인덱스)을 담고 있는 배열.
        num_classes (int): 총 클래스의 수. 원-핫 벡터의 길이가 됩니다.
        dtype (mx.Dtype, optional): 출력 텐서의 데이터 타입.

    Returns:
        mx.array: 원-핫 인코딩된 배열. shape은 (*indices.shape, num_classes)가 됩니다.
    """
    # broadcasting을 활용한 효율적인 one-hot 인코딩
    return (indices[..., None] == mx.arange(num_classes)).astype(dtype)