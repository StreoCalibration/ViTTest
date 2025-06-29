import torch
import os
from tqdm import tqdm
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data.aoi_dataset import AoiDataset
from data.transforms import get_transforms

# PyTorch 기본 collate 함수 (데이터로더에서 배치 생성을 위함)
def collate_fn(batch):
    return tuple(zip(*batch))

# 임시 모델 로더 (실제 프로젝트의 모델로 대체 가능)
def get_model_instance(num_classes):
    """Faster R-CNN 모델을 불러오는 예시 함수"""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """한 에포크 동안 모델을 학습합니다."""
    model.train()
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # 역전파
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=losses.item())

def run_training(config):
    """전체 학습 파이프라인을 실행합니다."""
    # 1. 설정 불러오기
    train_config = config['training']
    data_config = config['data']
    optim_config = config['optimizer']
    
    output_dir = train_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # 2. 장치 설정
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # 3. 데이터셋 및 데이터로더 준비
    dataset = AoiDataset(
        root_dir=data_config['root_dir'],
        data_source=data_config['data_source'],
        transform=get_transforms(is_train=True)
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn
    )

    # 4. 모델 준비 (클래스 수: 배경 + 결함 = 2)
    model = get_model_instance(num_classes=2) 
    model.to(device)

    # 5. 옵티마이저 준비
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=optim_config['lr'])

    # 6. 학습 루프 실행
    num_epochs = train_config.get('num_epochs', 10) # 설정 파일에 없으면 기본 10 에포크
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch)

    # 7. 최종 모델 저장
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    print("\n" + "="*42)
    print("✅ Training Finished! Model saved successfully.")
    print(f"   Model Path: {os.path.abspath(final_model_path)}")
    print("="*42)