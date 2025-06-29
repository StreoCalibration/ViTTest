import os
import random
from PIL import Image, ImageDraw

def make_random_defect(defect_size):
    """랜덤한 형태의 검은색 이물(결함) 이미지를 생성합니다."""
    defect = Image.new("RGBA", (defect_size, defect_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(defect)
    shape_type = random.choice(["ellipse", "rectangle", "polygon", "line"])
    color = (0, 0, 0, 255)  # 검은색

    if shape_type == "ellipse":
        # 타원형 이물
        bbox = [
            random.randint(0, defect_size // 4),
            random.randint(0, defect_size // 4),
            random.randint(3 * defect_size // 4, defect_size - 1),
            random.randint(3 * defect_size // 4, defect_size - 1)
        ]
        draw.ellipse(bbox, fill=color)
    elif shape_type == "rectangle":
        # 사각형 이물
        bbox = [
            random.randint(0, defect_size // 3),
            random.randint(0, defect_size // 3),
            random.randint(defect_size // 2, defect_size - 1),
            random.randint(defect_size // 2, defect_size - 1)
        ]
        draw.rectangle(bbox, fill=color)
    elif shape_type == "polygon":
        # 불규칙 다각형 이물
        num_points = random.randint(3, 7)
        points = [
            (
                random.randint(0, defect_size - 1),
                random.randint(0, defect_size - 1)
            )
            for _ in range(num_points)
        ]
        draw.polygon(points, fill=color)
    elif shape_type == "line":
        # 굵은 선 이물
        x1 = random.randint(0, defect_size - 1)
        y1 = random.randint(0, defect_size - 1)
        x2 = random.randint(0, defect_size - 1)
        y2 = random.randint(0, defect_size - 1)
        width = random.randint(defect_size // 10, defect_size // 4)
        draw.line((x1, y1, x2, y2), fill=color, width=width)

    return defect

def make_example_defect_template(real_data_dir, defect_template_dir, num_templates=5):
    # real_data 폴더에서 첫 번째 이미지를 찾음
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp')
    real_images = [f for f in os.listdir(real_data_dir) if f.lower().endswith(supported_formats)]
    if not real_images:
        print(f"real_data 폴더에 이미지가 없습니다: {real_data_dir}")
        return

    # 첫 번째 이미지를 열어서 결함 템플릿 크기 결정
    img_path = os.path.join(real_data_dir, real_images[0])
    img = Image.open(img_path).convert("RGBA")
    w, h = img.size
    defect_size = min(w, h) // 8

    os.makedirs(defect_template_dir, exist_ok=True)
    for i in range(num_templates):
        defect = make_random_defect(defect_size)
        defect_path = os.path.join(defect_template_dir, f"example_defect_{i+1}.png")
        defect.save(defect_path)
        print(f"예제 defect template 저장됨: {defect_path}")

if __name__ == "__main__":
    real_data_dir = "dataset_root/real_data"
    defect_template_dir = "dataset_root/defect_templates"
    make_example_defect_template(real_data_dir, defect_template_dir, num_templates=5)