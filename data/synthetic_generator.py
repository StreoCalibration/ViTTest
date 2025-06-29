import os
import glob
import random
import json
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm  # ← 이 줄을 추가하세요

class SyntheticDataGenerator:
    """
    배경 이미지에 결함 템플릿을 합성하여 가상 데이터를 생성합니다.
    설정 파일에 따라 다양한 증강(크기, 회전, 색상 등)을 적용하고,
    한 이미지에 여러 개의 결함을 합성할 수 있습니다.

    필요한 설정(config) 구조 예시:
    ```yaml
    source:
      base_image_dir: 'path/to/normal/images'
      defect_template_dir: 'path/to/defect/templates'
    output:
      image_dir: 'path/to/output/images'
      annotation_dir: 'path/to/output/annotations'
    generation:
      num_images: 1000
      max_defects_per_image: 3
      augmentation:
        scale_range: [0.05, 0.15]
        rotation_range: [0, 360]
        brightness_range: [0.8, 1.2]
        contrast_range: [0.8, 1.2]
    ```
    """
    def __init__(self, config: dict):
        """
        설정 파일로 생성기를 초기화합니다.

        Args:
            config (dict): 설정 정보를 담은 딕셔너리.
        """
        self.config = config
        self.generation_config = config['generation']
        self.output_config = config['output']
        self.source_config = config['source']

        # 증강 및 생성 관련 설정 불러오기 (기본값과 함께)
        aug_config = self.generation_config.get('augmentation', {})
        self.scale_range = aug_config.get('scale_range', [0.05, 0.15])
        self.rotation_range = aug_config.get('rotation_range', [0, 360])
        self.brightness_range = aug_config.get('brightness_range', [0.8, 1.2])
        self.contrast_range = aug_config.get('contrast_range', [0.8, 1.2])
        self.max_defects = self.generation_config.get('max_defects_per_image', 3)
        self.num_images = self.generation_config.get('num_images', 10)

        # 소스 이미지 경로 불러오기
        self.base_image_paths = self._get_image_paths(self.source_config['base_image_dir'])
        self.defect_template_paths = self._get_image_paths(self.source_config['defect_template_dir'])

        if not self.base_image_paths:
            raise FileNotFoundError(f"배경 이미지를 찾을 수 없습니다: {self.source_config['base_image_dir']}")
        if not self.defect_template_paths:
            raise FileNotFoundError(f"결함 템플릿을 찾을 수 없습니다: {self.source_config['defect_template_dir']}")

        # 출력 디렉토리 생성
        os.makedirs(self.output_config['image_dir'], exist_ok=True)
        os.makedirs(self.output_config['annotation_dir'], exist_ok=True)

    def _get_image_paths(self, directory: str) -> list:
        """디렉토리 및 하위 디렉토리에서 지원하는 모든 이미지 파일 경로를 재귀적으로 찾습니다."""
        supported_formats = ('png', 'jpg', 'jpeg', 'bmp')
        paths = []
        for fmt in supported_formats:
            # os.path.join을 사용하여 경로를 구성하고, recursive=True로 하위 디렉토리까지 탐색합니다.
            paths.extend(glob.glob(os.path.join(directory, f'**/*.{fmt}'), recursive=True))
        return paths

    def _apply_augmentations(self, defect_img: Image.Image, base_width: int) -> Image.Image:
        """결함 이미지에 다양한 증강을 적용합니다."""
        # 1. 크기 조절
        scale = random.uniform(*self.scale_range)
        new_width = int(base_width * scale)
        aspect_ratio = defect_img.height / defect_img.width
        defect_img = defect_img.resize((new_width, int(new_width * aspect_ratio)), resample=Image.BICUBIC)

        # 2. 색상/밝기/대비 조절
        enhancer = ImageEnhance.Brightness(defect_img)
        defect_img = enhancer.enhance(random.uniform(*self.brightness_range))
        enhancer = ImageEnhance.Contrast(defect_img)
        defect_img = enhancer.enhance(random.uniform(*self.contrast_range))

        # 3. 회전
        angle = random.uniform(*self.rotation_range)
        defect_img = defect_img.rotate(angle, expand=True, resample=Image.BICUBIC)

        return defect_img

    def random_transform(self, defect_img):
        # 무작위 회전, 크기 조절, 밝기/투명도 조절 등
        angle = random.randint(0, 359)
        scale = random.uniform(0.3, 1.0)
        alpha = random.uniform(0.5, 1.0)
        defect_img = defect_img.rotate(angle, expand=True)
        w, h = defect_img.size
        defect_img = defect_img.resize((int(w * scale), int(h * scale)))
        # 투명도 조절
        if defect_img.mode != 'RGBA':
            defect_img = defect_img.convert('RGBA')
        alpha_channel = defect_img.split()[-1].point(lambda p: int(p * alpha))
        defect_img.putalpha(alpha_channel)
        return defect_img

    def blend_defect(self, base_img, defect_img):
        # 결함 이미지를 base_img의 랜덤 위치에 합성
        bw, bh = base_img.size
        dw, dh = defect_img.size
        max_x = max(bw - dw, 1)
        max_y = max(bh - dh, 1)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        base_img.paste(defect_img, (x, y), defect_img)
        bbox = [x, y, x + dw, y + dh]
        return base_img, bbox

    def generate(self):
        """
        메인 생성 루프. 설정된 수량만큼 가상 이미지와 어노테이션을 생성합니다.
        """
        print(f"{self.num_images}개의 가상 데이터 생성을 시작합니다...")

        for i in tqdm(range(self.num_images), desc="데이터 생성 중"):
            # 1. 무작위로 배경 이미지 선택 및 로드
            base_path = random.choice(self.base_image_paths)
            base_img = Image.open(base_path).convert("RGBA")

            # 한 이미지에 대한 결함 정보들을 저장할 리스트
            defects_in_image = []
            num_defects_to_add = random.randint(1, self.max_defects)

            for _ in range(num_defects_to_add):
                # 2. 결함 템플릿 선택 및 증강 적용
                defect_path = random.choice(self.defect_template_paths)
                defect_img = Image.open(defect_path).convert("RGBA")
                
                augmented_defect = self._apply_augmentations(defect_img, base_img.width)

                # 3. 배경 이미지의 유효한 위치에 결함 배치
                # 회전 등으로 인해 결함이 배경보다 커지는 경우 방지
                max_x = base_img.width - augmented_defect.width
                max_y = base_img.height - augmented_defect.height
                
                if max_x < 0 or max_y < 0:
                    # 증강된 결함이 너무 커서 배경에 들어갈 수 없으면 건너뜀
                    continue

                pos_x = random.randint(0, max_x)
                pos_y = random.randint(0, max_y)

                # 4. 알파 채널을 이용해 두 이미지 합성
                base_img.paste(augmented_defect, (pos_x, pos_y), augmented_defect)

                # 5. 어노테이션 정보 생성
                bbox = [pos_x, pos_y, pos_x + augmented_defect.width, pos_y + augmented_defect.height]
                defect_info = {
                    "class_name": os.path.splitext(os.path.basename(defect_path))[0],
                    "bbox": bbox
                }
                defects_in_image.append(defect_info)

            # 모든 결함이 합성된 최종 이미지
            synthetic_img = base_img.convert("RGB")

            # 6. 최종 어노테이션 파일 생성
            image_filename = f"synthetic_{i:05d}.jpg"
            annotation = {
                "image_name": image_filename,
                "defects": defects_in_image
            }

            # 7. 이미지와 어노테이션 파일 저장
            annotation_filename = f"synthetic_{i:05d}.json"
            synthetic_img.save(os.path.join(self.output_config['image_dir'], image_filename), 'JPEG')
            
            # 생성된 결함이 하나도 없는 경우 빈 json 파일 생성 방지
            if defects_in_image:
                with open(os.path.join(self.output_config['annotation_dir'], annotation_filename), 'w', encoding='utf-8') as f:
                    json.dump(annotation, f, indent=4, ensure_ascii=False)

        print(f"\n성공적으로 {self.num_images}개의 이미지와 어노테이션을 생성했습니다.")
        print(f"이미지 저장 경로: {self.output_config['image_dir']}")
        print(f"어노테이션 저장 경로: {self.output_config['annotation_dir']}")