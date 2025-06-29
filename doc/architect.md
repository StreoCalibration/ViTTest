## 2. 4+1 View 아키텍처 설계 (v0.2)

### 2.1. 유스케이스 뷰 (Use Case View)

*기존 UC-01 ~ 04는 유지되며, 새로운 요구사항에 따라 UC-05가 추가되고 UC-02가 확장됩니다.*

* **액터 (Actors):**
    * 시스템 운영자
    * 머신러닝 엔지니어

* **유스케이스 (Use Cases):**
    * **(UC-01)** 이물 검사 수행 (변경 없음)
    * **(UC-02) 모델 학습 및 평가 (실제/가상 데이터):** 머신러닝 엔지니어는 **실제 데이터셋, 가상으로 생성된 데이터셋, 또는 둘을 혼합한 데이터셋**을 사용하여 이물 검사 모델을 학습시키고 성능을 평가합니다.
    * **(UC-03)** 결과 시각화 (변경 없음)
    * **(UC-04)** 데이터 관리 (변경 없음)
    * **(UC-05) 가상 데이터 생성:** 머신러닝 엔지니어는 정상 제품 이미지에 다양한 형태의 가상 이물을 합성하여 학습용 데이터를 대량으로 생성합니다. 이 프로세스는 이물의 종류, 크기, 위치, 투명도 등을 무작위로 조절하여 다양한 시나리오에 대한 학습을 가능하게 합니다.

### 2.2. 논리 뷰 (Logical View)

*새로운 요구사항을 반영하여 `SyntheticDataGenerator` 컴포넌트가 추가되고, `PredictionHeadModule` 및 `LossModule`이 DETR 아키텍처를 기반으로 재정의됩니다.*

* **주요 컴포넌트:**
    * **`SyntheticDataGenerator`**: **(신규)**
        * **책임:** 정상 제품 이미지와 이물 템플릿 이미지를 입력받아, 다양한 변형(회전, 크기 조절, 색상 변경 등)을 적용한 후, 정상 이미지 위에 자연스럽게 합성(blending)합니다. 합성된 이물의 위치 정보를 기반으로 바운딩 박스 또는 세그멘테이션 마스크 형태의 어노테이션을 자동으로 생성합니다.
        * **구현:**  
            - `data/synthetic_generator.py`에 구현되어 있으며, 설정 파일(`.yaml`)을 받아 배경 이미지와 결함 템플릿을 불러와 다양한 증강(크기, 회전, 밝기, 대비 등)을 적용합니다.
            - 한 이미지에 여러 개의 결함을 랜덤하게 합성할 수 있으며, 결함의 형태도 랜덤하게 생성할 수 있습니다(예: 타원, 사각형, 다각형, 선 등 다양한 검은색 이물).
            - 합성된 이미지와 바운딩 박스 어노테이션(JSON)을 자동으로 저장합니다.
        * **인터페이스:** `generate(base_image, defect_template, config)`

    * **`DataPreprocessingModule`**:
        * **책임:** **실제 데이터와 가상 데이터를 모두 처리**할 수 있도록 확장됩니다. 이미지 로드, 패치 분할, 정규화 등 기존 역할은 동일합니다.

    * **`AugmentationModule`**: (변경 없음)

    * **`VisionBackbone (SwinTransformerV2)`**:
        * **책임:** 이미지 패치 시퀀스로부터 특징을 추출합니다. 이 접근 방식은 전통적인 CNN이 **가중치 공유(weight sharing)**를 통해 강제하는 강력한 **공간적 유도 편향(inductive bias)**과 달리, 데이터로부터 직접 전역적인 관계를 학습하는 데 중점을 둡니다.

    * **`PredictionHeadModule (DETR-style)`**: **(상세화)**
        * [cite_start]**책임:** DETR 아키텍처를 참조하여 Transformer Decoder와 FFN(Feed Forward Network)으로 구성됩니다[cite: 767, 867]. [cite_start]Decoder는 고정된 수(N)의 학습 가능한 **'Object Query'**를 입력으로 받습니다[cite: 768, 888]. [cite_start]이 쿼리들은 이미지 전체 컨텍스트(Encoder 출력)에 어텐션하여, 각 쿼리에 해당하는 객체의 클래스와 위치를 병렬적으로 직접 예측합니다[cite: 768, 884].
        * **인터페이스:** `predict(encoder_features, object_queries)`

    * **`LossModule (Hungarian Matcher)`**: **(상세화)**
        * [cite_start]**책임:** DETR에서 제안된 **이분 매칭(Bipartite Matching) 기반의 Hungarian Loss**를 채택합니다[cite: 767]. [cite_start]이 손실은 N개의 예측과 M개의 정답 객체 간의 고유한 일대일 매칭을 강제하여 중복 예측 문제를 근본적으로 해결합니다[cite: 852].
        * **세부 구성:**
            1.  [cite_start]**매칭 비용(Matching Cost):** 클래스 예측 확률과 예측/정답 박스 간의 유사도(L1 Loss + Generalized IoU Loss)를 결합하여 계산합니다[cite: 851, 865].
            2.  [cite_start]**헝가리안 알고리즘(Hungarian Algorithm):** 매칭 비용을 최소화하는 최적의 할당(assignment)을 찾습니다[cite: 847].
            3.  [cite_start]**최종 손실 계산:** 할당된 쌍에 대해서만 클래스 손실(Negative Log-Likelihood)과 박스 손실을 계산하여 합산합니다[cite: 854].

    * **`TrainingOrchestrator`**: (변경 없음)

    * **`InferenceEngine`**:
        * **책임:** 학습된 모델을 사용한 추론을 수행합니다. [cite_start]DETR 기반 헤드를 사용하므로, **NMS와 같은 후처리 과정이 필요 없어 파이프라인이 단순화됩니다**[cite: 766].

![Logical View Diagram v0.2](https://i.imgur.com/k2p8zCq.png)

### 2.3. 개발 뷰 (Development View)

*가상 데이터 생성 모듈이 추가되고, 데이터셋 디렉토리 구조가 구체화됩니다.*

* **기본 언어:** Python
* [cite_start]**주요 라이브러리:** PyTorch, torchvision, mmcv, mmdetection [cite: 1464][cite_start], mmsegmentation [cite: 1471], Pillow, tqdm
* **제안 패키지 구조 (v0.2):**
    ```python
    swin_aoi_system/
    ├── configs/                 # 모델, 학습, 데이터 관련 설정 파일 (e.g., yaml)
    │   ├── model/swin_v2_base.yaml
    │   └── training/train_config.yaml
    ├── data/                    # 데이터셋 로딩 및 전처리
    │   ├── __init__.py
    │   ├── aoi_dataset.py       # PyTorch Dataset 클래스 (실제/가상 데이터 로드)
    │   ├── transforms.py        # 이미지 증강 및 변환 로직
    │   ├── synthetic_generator.py # (신규) SyntheticDataGenerator 모듈
    │   └── generate_synthetic_data.py # (신규) 가상 데이터 생성 실행 스크립트
    ├── models/                  # 모델 아키텍처
    │   ├── __init__.py
    │   ├── backbone_swin_v2.py  # Swin Transformer V2 백본 구현
    │   ├── attention.py         # Scaled Cosine Attention 등 어텐션 모듈
    │   ├── detection_head_detr.py # (구체화) DETR 스타일 Decoder 및 FFN 헤드
    │   └── layers.py            # PatchMerging, MLP 등 기본 레이어
    ├── engine/                  # 학습 및 추론 엔진
    │   ├── __init__.py
    │   ├── trainer.py           # 학습 루프 및 오케스트레이션
    │   ├── predictor.py         # 추론 로직
    │   └── losses.py            # (구체화) Hungarian Matcher 및 관련 손실 함수
    ├── utils/                   # 보조 유틸리티
    │   ├── __init__.py
    │   ├── gpu_optimizer.py     # ZeRO, Activation Checkpointing 등 메모리 최적화 유틸
    │   ├── scheduler.py         # 학습률 스케줄러
    │   └── visualizer.py        # 결과 시각화 유틸
    ├── dataset_root/            # (신규) 데이터 루트 디렉토리
    │   ├── real_data/           # 실제 촬영된 이미지 및 어노테이션
    │   ├── defect_templates/    # (신규) 다양한 형태의 이물 템플릿 이미지 (랜덤 생성 가능)
    │   └── synthetic_data/      # 가상으로 생성된 이미지 및 어노테이션
    ├── train.py                 # 모델 학습 스크립트
    ├── evaluate.py              # 모델 평가 스크립트
    └── inference.py             # 단일 이미지 추론 스크립트
    ```

### 2.4. 프로세스 뷰 (Process View)

*가상 데이터 생성 프로세스가 추가되고, 학습/테스트 프로세스가 이를 활용하도록 수정됩니다.*

* **가상 데이터 생성 프로세스:**
    1.  `generate_synthetic_data.py` 스크립트가 실행됩니다.
    2.  스크립트는 `dataset_root/real_data/`에서 배경으로 사용할 정상 제품 이미지를 로드합니다.
    3.  사전에 정의된 이물 템플릿 풀(`dataset_root/defect_templates/`)에서 이물을 무작위로 선택하거나, 랜덤 생성 로직을 통해 다양한 형태의 검은색 이물(타원, 사각형, 다각형, 선 등)을 자동 생성할 수 있습니다.
    4.  `SyntheticDataGenerator`가 이물에 변형(크기, 회전, 밝기, 대비 등)을 가한 후 정상 이미지에 합성하고, 어노테이션을 생성합니다.
    5.  생성된 이미지와 어노테이션을 `dataset_root/synthetic_data/`에 저장합니다.
    6.  지정된 수량만큼 2-5단계를 반복합니다.

* **학습 프로세스:**
    1.  `train.py` 실행 시, 설정 파일(`configs/`)을 통해 **사용할 데이터 소스(실제, 가상, 혼합)를 지정**합니다.
    2.  `aoi_dataset.py`는 지정된 소스에서 데이터를 로드합니다.
    3.  (이후 과정은 v0.1과 동일)

* **테스트(추론) 프로세스:**
    1.  `evaluate.py` 또는 `inference.py` 실행 시, 테스트할 데이터(실제 또는 가상)를 지정합니다.
    2.  (이후 과정은 v0.1과 동일)

### 2.5. 물리 뷰 (Physical View)

*하드웨어 및 소프트웨어 요구사항은 v0.1과 동일하며, 안정적인 설계를 유지합니다.*

* **학습 환경:**
    * **하드웨어:** 다중 GPU 서버 (예: 8 x NVIDIA V100 또는 A100 GPU), 고속 스토리지(SSD/NVMe)
    * **소프트웨어:** Linux OS (Ubuntu), Python, PyTorch, CUDA, cuDNN, NCCL (다중 GPU 통신용)

* **배포(추론) 환경:**
    * **하드웨어:** AOI 장비에 내장된 산업용 PC. 실시간 처리 요구사항에 따라 고성능 GPU(예: NVIDIA RTX 시리즈) 1대 탑재.
    * **소프트웨어:** Linux OS 또는 Windows, 최적화된 추론 라이브러리(예: TensorRT)를 포함한 Python/C++ 런타임 환경.

### 2.6. 실행 가이드 (Execution Guide)

#### 2.6.1. 가상 데이터 생성

`generate_synthetic_data.py` 스크립트를 사용하여 학습에 필요한 가상 데이터를 생성할 수 있습니다.

**1. 사전 준비:**

*   **배경 이미지 준비:** `dataset_root/real_data/` 디렉토리에 배경으로 사용할 정상 제품 이미지들을 준비합니다.
*   **결함 템플릿 준비:** `dataset_root/defect_templates/` 디렉토리에 합성할 이물/결함 템플릿 이미지(배경이 투명한 `.png` 파일 권장)를 준비하거나, 랜덤 이물 생성 스크립트(`make_example_defect_template.py`)를 실행하여 다양한 형태의 검은색 이물 템플릿을 자동 생성할 수 있습니다.
*   **설정 파일 수정:** `configs/generation/synthetic_data_config.yaml` 파일을 열어 `base_image_dir`와 `defect_template_dir` 경로가 올바른지 확인하고, 필요에 따라 생성할 이미지 수(`num_images`) 등의 옵션을 수정합니다.

**2. 생성 명령어 실행:**

프로젝트의 루트 디렉토리에서 다음 명령어를 실행합니다.

```bash
python -m swin_aoi_system.generate_synthetic_data --config swin_aoi_system/configs/generation/synthetic_data_config.yaml
```

**3. 결과 확인:**

생성이 완료되면 `output` 설정에 지정된 경로(기본값: `dataset_root/synthetic_data/`)에서 생성된 이미지와 JSON 어노테이션 파일을 확인할 수 있습니다.

**4. (선택) 랜덤 이물 템플릿 자동 생성:**

다양한 형태의 검은색 이물 템플릿을 자동으로 생성하려면 아래 스크립트를 실행하세요.

```bash
python make_example_defect_template.py
```

이 스크립트는 `dataset_root/defect_templates/` 폴더에 타원, 사각형, 다각형, 선 등 다양한 형태의 검은색 이물 PNG 파일을 생성합니다.