# 제1회 퀀텀 AI 경진대회 베이스라인 솔루션

이 프로젝트는 Fashion MNIST 데이터셋을 사용하여 이미지를 분류하는 양자 머신 러닝(QML) 모델을 구현하고, 대회 우승을 목표로 성능을 최적화하는 과정을 담고 있습니다. Pennylane과 PyTorch를 결합한 하이브리드 양자-고전 신경망을 사용하여 이미지 분류 문제를 해결합니다.

## 목차

- [제1회 퀀텀 AI 경진대회 베이스라인 솔루션](#제1회-퀀텀-ai-경진대회-베이스라인-솔루션)
  - [목차](#목차)
  - [프로젝트 개요](#프로젝트-개요)
  - [최종 성능](#최종-성능)
  - [양자 머신 러닝(QML) 소개](#양자-머신-러닝qml-소개)
  - [실행 환경 설정](#실행-환경-설정)
    - [방법 1: uv를 사용한 가상환경 설정 (권장)](#방법-1-uv를-사용한-가상환경-설정-권장)
    - [방법 2: 기존 pip 사용](#방법-2-기존-pip-사용)
    - [Jupyter Notebook 실행](#jupyter-notebook-실행)
  - [코드 설명](#코드-설명)
  - [모델 개선 가이드](#모델-개선-가이드)

## 프로젝트 개요

본 프로젝트는 Fashion MNIST 데이터셋의 'T-shirt/top'(레이블 0)과 'Coat'(레이블 6) 두 가지 클래스를 분류하는 이진 분류 모델을 개발합니다. 모델은 다음과 같은 하이브리드 구조를 가집니다.

-   **고전적 부분 (Classical Part)**: 합성곱 신경망(CNN)을 사용하여 입력 이미지의 특징을 추출합니다.
-   **양자 부분 (Quantum Part)**: 양자 회로(QNN)를 사용하여 CNN에서 추출된 특징을 처리하고 최종 분류 결과를 출력합니다.

## 최종 성능

-   **테스트 정확도 (레이블 0과 6 대상)**: **82.85%**

## 양자 머신 러닝(QML) 소개

양자 머신 러닝은 양자 컴퓨팅의 원리를 머신 러닝에 적용하여 기존의 고전적인 방법으로는 해결하기 어려운 문제에 대한 새로운 접근법을 제시하는 분야입니다.

-   **양자 비트 (Qubit)**: 0과 1의 상태를 동시에 가질 수 있는 **중첩(superposition)** 상태를 이용하여 고전 비트보다 훨씬 많은 정보를 표현합니다.
-   **양자 게이트 (Quantum Gate)**: 큐빗의 상태를 조작하는 연산으로, 중첩과 **얽힘(entanglement)** 같은 양자 현상을 활용합니다.
-   **양자 회로 (Quantum Circuit)**: 학습 가능한 파라미터를 가진 양자 게이트의 조합으로, 머신 러닝 모델의 일부로 사용될 수 있습니다.

## 실행 환경 설정

이 프로젝트를 실행하기 위해서는 다음 라이브러리들이 필요합니다.

### 방법 1: uv를 사용한 가상환경 설정 (권장)

[uv](https://docs.astral.sh/uv/)는 빠르고 현대적인 Python 패키지 및 프로젝트 관리자입니다.

```bash
# uv 설치 (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 가상환경 생성 및 활성화
uv venv
source .venv/bin/activate  # macOS/Linux
# 또는 Windows의 경우: .venv\Scripts\activate

# 필요한 패키지 설치
uv pip install torch torchvision pennylane matplotlib tqdm jupyter notebook jupyterlab
```

### 방법 2: 기존 pip 사용

```bash
# 가상환경 생성 (선택사항)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 또는 Windows의 경우: .venv\Scripts\activate

# 패키지 설치
pip install torch torchvision pennylane matplotlib tqdm jupyter notebook jupyterlab
```

### Jupyter Notebook 실행

환경 설정 후 다음 명령어로 Jupyter Notebook을 실행할 수 있습니다:

```bash
jupyter notebook
# 또는 JupyterLab 사용
jupyter lab
```

## 코드 설명

주요 코드는 `main.py` 스크립트에 포함되어 있습니다. 코드의 각 부분은 다음과 같은 역할을 수행합니다.

-   **데이터 로딩 및 전처리**: `torchvision`을 사용하여 Fashion MNIST 데이터셋을 로드하고, 이진 분류를 위해 특정 클래스만 필터링합니다.
-   **하이브리드 모델 정의**: `BinaryClassifier` 클래스는 CNN 레이어와 Pennylane의 양자 회로(`qnn`)를 결합하여 정의됩니다.
-   **학습 루프**: `Adam` 옵티마이저와 `NLLLoss` 손실 함수를 사용하여 모델을 학습시킵니다.
-   **추론 및 평가**: 학습된 모델을 사용하여 테스트 데이터셋의 성능을 평가하고, 결과를 `.csv` 파일로 저장합니다.

## 모델 개선 가이드

모델의 성능을 더욱 향상시키기 위해 다음과 같은 방법을 시도해볼 수 있습니다.

-   **다양한 양자 회로 실험**:
    -   `qml.StronglyEntanglingLayers`나 `qml.BasicEntanglerLayers`와 같은 고수준의 템플릿을 사용하여 더 복잡한 양자 회로를 구성해볼 수 있습니다.
    -   회로의 깊이(depth)나 파라미터 수를 조절하여 모델의 표현력과 과적합 사이의 균형을 맞출 수 있습니다.

-   **고전 레이어 변경**:
    -   CNN 레이어의 필터 수, 커널 크기, 또는 레이어 수를 변경하여 특징 추출 성능을 조절할 수 있습니다.
    -   FC 레이어의 뉴런 수를 조정하여 양자 회로로 전달되는 특징 벡터의 차원을 변경해볼 수 있습니다.

-   **데이터 인코딩 방식 탐구**:
    -   현재 사용된 `RZ` 게이트 기반의 인코딩 외에, `qml.AngleEmbedding`이나 `qml.AmplitudeEmbedding` 등 다른 데이터 인코딩 전략을 시도해볼 수 있습니다. 데이터 인코딩 방식은 모델 성능에 큰 영향을 미칠 수 있습니다.

-   **옵티마이저 및 학습률 스케줄러**:
    -   `Adam` 외에 `AdamW`, `SGD` 등 다른 옵티마이저를 테스트해볼 수 있습니다.
    -   `torch.optim.lr_scheduler`를 사용하여 학습 과정 동안 학습률을 동적으로 조절하면 더 안정적이고 빠른 수렴을 유도할 수 있습니다.

---
*이 README 파일은 퀀텀 AI 경진대회 참가를 위해 제작되었습니다.*
