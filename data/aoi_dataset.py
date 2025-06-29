import os
import json
import torch
from PIL import Image
import glob

class AoiDataset(torch.utils.data.Dataset):
    """
    실제 및 가상 AOI 데이터를 로드하기 위한 PyTorch Dataset 클래스.
    'mixed', 'real', 'synthetic' 소스 타입을 지원합니다.
    """
    def __init__(self, root_dir, data_source='mixed', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # 1. 설정에 따라 이미지 경로 수집
        if data_source in ['real', 'mixed']:
            real_image_dir = os.path.join(self.root_dir, 'real_data')
            if os.path.isdir(real_image_dir):
                self.image_paths.extend(self._get_image_paths(real_image_dir))

        if data_source in ['synthetic', 'mixed']:
            synthetic_image_dir = os.path.join(self.root_dir, 'synthetic_data', 'images')
            if os.path.isdir(synthetic_image_dir):
                self.image_paths.extend(self._get_image_paths(synthetic_image_dir))

        if not self.image_paths:
            raise FileNotFoundError(f"데이터 디렉토리에서 이미지를 찾을 수 없습니다. 경로를 확인하세요: {os.path.abspath(root_dir)}")

    def _get_image_paths(self, directory):
        """디렉토리 내 모든 이미지 파일 경로를 재귀적으로 찾습니다."""
        supported_formats = ('png', 'jpg', 'jpeg', 'bmp')
        paths = []
        for fmt in supported_formats:
            paths.extend(glob.glob(os.path.join(directory, f'**/*.{fmt}'), recursive=True))
        return paths

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        # 2. 어노테이션 파일 경로를 찾아 파싱 (가상 데이터에만 해당)
        if 'synthetic_data' in img_path.replace('\\', '/'):
            ann_dir = os.path.join(self.root_dir, 'synthetic_data', 'annotations')
            img_filename = os.path.basename(img_path)
            ann_filename = os.path.splitext(img_filename)[0] + '.json'
            ann_path = os.path.join(ann_dir, ann_filename)

            if os.path.exists(ann_path):
                with open(ann_path, 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)
                for defect in ann_data.get('defects', []):
                    boxes.append(defect['bbox'])
                    labels.append(1) # 단일 클래스 'defect'를 1로 가정

        # 3. 타겟 텐서 생성 (오류 수정 지점)
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        # 바운딩 박스가 없는 경우, 텐서 형태를 [0, 4]로 명시적으로 지정
        if boxes_tensor.numel() == 0:
            boxes_tensor = boxes_tensor.reshape(0, 4)

        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        
        # 모델이 요구하는 추가 정보 생성
        target["area"] = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)

        if self.transform:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.image_paths)