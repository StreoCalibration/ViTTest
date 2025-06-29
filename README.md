# ViTTest

## 실행 가이드

### 1. 가상 데이터 생성

#### 1.1. 사전 준비

- **배경 이미지 준비:**  
  `dataset_root/real_data/` 디렉토리에 배경으로 사용할 정상 제품 이미지들을 준비합니다.
- **결함 템플릿 준비:**  
  `dataset_root/defect_templates/` 디렉토리에 합성할 이물/결함 템플릿 이미지(배경이 투명한 `.png` 파일 권장)를 준비하거나, 랜덤 이물 생성 스크립트(`make_example_defect_template.py`)를 실행하여 다양한 형태의 검은색 이물 템플릿을 자동 생성할 수 있습니다.
- **설정 파일 수정:**  
  `configs/generation/synthetic_data_config.yaml` 파일을 열어 `base_image_dir`와 `defect_template_dir` 경로가 올바른지 확인하고, 필요에 따라 생성할 이미지 수(`num_images`) 등의 옵션을 수정합니다.

#### 1.2. 생성 명령어 실행

프로젝트의 루트 디렉토리(`swin_aoi_system/`)에서 다음 명령어를 실행합니다.

```bash
python generate_synthetic_data.py --config configs/generation/synthetic_data_config.yaml
```

#### 1.3. 결과 확인

생성이 완료되면 `output` 설정에 지정된 경로(기본값: `dataset_root/synthetic_data/`)에서 생성된 이미지와 JSON 어노테이션 파일을 확인할 수 있습니다.


### 2. 실행 가이드 (Execution Guide)

#### 2.1. 가상 데이터 생성

`generate_synthetic_data.py` 스크립트를 사용하여 학습에 필요한 가상 데이터를 생성할 수 있습니다.

**1. 사전 준비:**

*   **배경 이미지 준비:** `dataset_root/real_data/` 디렉토리에 배경으로 사용할 정상 제품 이미지들을 준비합니다.
*   **결함 템플릿 준비:** `dataset_root/defect_templates/` 디렉토리에 합성할 이물/결함 템플릿 이미지(배경이 투명한 `.png` 파일 권장)를 준비하거나, 랜덤 이물 생성 스크립트(`make_example_defect_template.py`)를 실행하여 다양한 형태의 검은색 이물 템플릿을 자동 생성할 수 있습니다.
*   **설정 파일 수정:** `configs/generation/synthetic_data_config.yaml` 파일을 열어 `base_image_dir`와 `defect_template_dir` 경로가 올바른지 확인하고, 필요에 따라 생성할 이미지 수(`num_images`) 등의 옵션을 수정합니다.

**2. 생성 명령어 실행:**

프로젝트의 루트 디렉토리(`swin_aoi_system/`)에서 다음 명령어를 실행합니다.

```bash
python 1_1_generate_synthetic_data.py --config configs/generation/synthetic_data_config.yaml
```

**3. 결과 확인:**

생성이 완료되면 `output` 설정에 지정된 경로(기본값: `dataset_root/synthetic_data/`)에서 생성된 이미지와 JSON 어노테이션 파일을 확인할 수 있습니다.

**4. (선택) 랜덤 이물 템플릿 자동 생성:**

다양한 형태의 검은색 이물 템플릿을 자동으로 생성하려면 아래 스크립트를 실행하세요.

```bash
python make_example_defect_template.py
```

이 스크립트는 `dataset_root/defect_templates/` 폴더에 타원, 사각형, 다각형, 선 등 다양한 형태의 검은색 이물 PNG 파일을 생성합니다.

python -m swin_aoi_system.train --config swin_aoi_system\configs\training\train_config.yaml