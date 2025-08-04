# OpenMLXModel

## 목적
해당 저장소는 학습 가능한 모델 코드를 만드는 것에 더 중점을 둡니다. 사용자는 [MLX](https://github.com/ml-explore/mlx)로 작성된 다양한 LLM 콤포넌트를 이해하고, 원하는 대로 조합하는 것으로 LLM에 대한 이해를 높일 수 있습니다. Mac OS에서는 학습 실습과 연습을 수행하고, 실질적으로 대규모 학습까지 가능하게 합니다.

## 구현되어 있는 것
### bfloat16 및 float32 기반의 학습
float16은 형변환이 많이 일어나 실질적으로 학습 효율이 좋지 않습니다. float8의 경우 mlx에서 지원할 예정이 아직 없으므로, 이를 염두해 두어야 합니다. [MLX 공식 문서](https://ml-explore.github.io/mlx/build/html/index.html) 에서는 기본적으로 float32 기반의 연산을 수행하며, bfloat16으로 변환하더라도 메모리 사용량이 급격하게 줄어드는 것은 아니라는 것을 기억하여야 합니다.

## 제공 예시
[examples/LLM](https://github.com/Nipol/OpenMLXModel/tree/main/examples/LLM)
[examples/GatelessMoELLM](https://github.com/Nipol/OpenMLXModel/tree/main/examples/GatelessMoELLM)
