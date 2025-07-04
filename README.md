# Swin AOI System

본 프로젝트는 Swin Transformer V2 모델을 기반으로 하는 자동 광학 검사(AOI) 시스템입니다.
실제 데이터와 가상으로 생성된 데이터를 모두 활용하여 이물 검출 모델을 학습하고 평가하는 기능을 제공합니다.

## 🚀 실행 가이드 (Execution Guide)

모든 명령어는 프로젝트의 루트 디렉토리(`F:\Source\ViTTest`)에서 실행하는 것을 기준으로 합니다.

### 1. 초기 설정 (Initial Setup)

1.  **데이터 디렉토리 준비**

    프로젝트를 실행하기 전에 아래와 같은 디렉토리 구조를 준비해야 합니다. `dataset_root` 폴더가 없다면 생성해주세요.

    ```
    dataset_root/
    ├── real_data/
    └── defect_templates/
    ```

    *   `dataset_root/real_data/`: 배경으로 사용할 정상 제품 이미지들을 이곳에 위치시킵니다.
    *   `dataset_root/defect_templates/`: 합성에 사용할 결함/이물 템플릿 이미지(배경이 투명한 `.png` 파일 권장)들을 이곳에 위치시킵니다.

2.  **(선택) 예제 결함 템플릿 생성**

    만약 사용할 결함 템플릿이 없다면, 아래 명령어를 실행하여 예제 템플릿(타원, 사각형, 선 등)을 자동으로 생성할 수 있습니다.

    ```bash
    python -m make_example_defect_template
    ```

### 2. 가상 데이터 생성 (Synthetic Data Generation)

정상 이미지와 결함 템플릿을 합성하여 학습에 사용할 가상 데이터를 생성합니다. **이때 모듈 이름에 `.py`를 붙이지 않도록 주의하세요.**

```bash
python -m generate_synthetic_data --config configs/generation/synthetic_data_config.yaml
```
*   생성된 데이터는 `dataset_root/synthetic_data/`에 저장됩니다.
*   필요 시 `python 0_clear_synthetic_data.py` 명령으로 기존 가상 데이터를 모두 삭제할 수 있습니다.

### 3. 모델 학습 (Model Training)

준비된 데이터를 사용하여 모델을 학습합니다.

```bash
python -m train --config configs/training/train_config.yaml
```

### 4. 모델 평가 (Model Evaluation)

학습이 끝난 후에는 평가 스크립트를 사용하여 모델의 성능을 확인할 수 있습니다.

```bash
python evaluate.py --model-path outputs/training_results/final_model.pth --data-root dataset_root
```

`--data-root` 인자는 `real_data`와 `synthetic_data` 폴더를 포함하는 최상위 디렉터리(`dataset_root`)를 가리켜야 합니다. 
잘못하여 `dataset_root/synthetic_data/images` 경로를 지정하면 데이터가 없다는 오류가 발생하므로 주의하세요.

필요에 따라 `--data-source`, `--iou-threshold`, `--score-threshold` 등의 옵션을 조정하여 평가 기준을 변경할 수 있습니다.

평가 결과로 Precision, Recall, F1-score가 출력됩니다.
