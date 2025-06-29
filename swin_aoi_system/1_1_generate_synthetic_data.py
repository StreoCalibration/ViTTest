import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 경로 문제를 방지하기 위해 os 모듈을 사용합니다.
import argparse
import os
import yaml
# data 패키지에서 SyntheticDataGenerator 클래스를 임포트합니다.
import traceback
import glob
from swin_aoi_system.data.synthetic_generator import SyntheticDataGenerator

def main(args):
    """Main function to generate synthetic data."""
    # 1. Load config from the specified YAML file
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        return
    except Exception as e:
        print(f"Error loading or parsing config file: {e}")
        return

    # 2. Convert relative paths to absolute paths based on this file location
    script_dir = os.path.dirname(__file__)
    for key in ("base_image_dir", "defect_template_dir"):
        path = config.get("source", {}).get(key)
        if path and not os.path.isabs(path):
            config["source"][key] = os.path.join(script_dir, path)

    for key in ("image_dir", "annotation_dir"):
        path = config.get("output", {}).get(key)
        if path and not os.path.isabs(path):
            config["output"][key] = os.path.join(script_dir, path)

    # 3. Override config values with command-line arguments if provided
    if args.num_images is not None:
        config['generation']['num_images'] = args.num_images

    if args.output_dir is not None:
        # The output_dir is the root for images and annotations
        config['output']['image_dir'] = os.path.join(args.output_dir, 'images')
        config['output']['annotation_dir'] = os.path.join(args.output_dir, 'annotations')

    print("Generating synthetic data with the following configuration:")
    print(yaml.dump(config, indent=2))

    def _has_image_files(directory):
        """디렉토리 내에 지원하는 이미지 파일이 있는지 확인합니다."""
        if not os.path.isdir(directory):
            return False
        supported_formats = ('png', 'jpg', 'jpeg', 'bmp')
        for fmt in supported_formats:
            if glob.glob(os.path.join(directory, f'**/*.{fmt}'), recursive=True):
                return True
        return False

    # 3. 데이터 생성 전 필수 경로 확인
    source_config = config.get('source', {})
    base_image_dir = source_config.get('base_image_dir')
    defect_template_dir = source_config.get('defect_template_dir')

    if not base_image_dir or not os.path.isdir(base_image_dir):
        print(f"\n[오류] 설정 파일에 명시된 배경 이미지 디렉토리를 찾을 수 없습니다: '{base_image_dir}'")
        print("`configs/generation/synthetic_data_config.yaml` 파일의 `source.base_image_dir` 경로를 확인해주세요.")
        return

    if not _has_image_files(base_image_dir):
        print(f"\n[오류] 배경 이미지 디렉토리 '{base_image_dir}'가 비어 있거나 지원하는 이미지 파일(.png, .jpg 등)이 없습니다.")
        print("해당 디렉토리에 배경으로 사용할 이미지를 추가해주세요.")
        return

    if not defect_template_dir or not os.path.isdir(defect_template_dir):
        print(f"\n[오류] 설정 파일에 명시된 결함 템플릿 디렉토리를 찾을 수 없습니다: '{defect_template_dir}'")
        print("`configs/generation/synthetic_data_config.yaml` 파일의 `source.defect_template_dir` 경로를 확인해주세요.")
        return

    if not _has_image_files(defect_template_dir):
        print(f"\n[오류] 결함 템플릿 디렉토리 '{defect_template_dir}'가 비어 있거나 지원하는 이미지 파일(.png, .jpg 등)이 없습니다.")
        print("해당 디렉토리에 합성할 결함 템플릿 이미지를 추가해주세요.")
        return


    # 4. SyntheticDataGenerator를 초기화하고 데이터 생성을 실행합니다.
    try:
        generator = SyntheticDataGenerator(config)
        generator.generate()
    except Exception as e:
        print(f"\n데이터 생성 중 오류가 발생했습니다: {e}")
        print("설정 파일의 경로, 원본 데이터의 존재 여부를 확인해주세요.")
        print("--- 상세 오류 정보 ---")
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic AOI data.")
    parser.add_argument('--config', type=str, required=True, help='Path to the generation configuration YAML file.')
    parser.add_argument('--output-dir', type=str, help='Path to the root output directory. Overrides config values.')
    parser.add_argument('--num-images', type=int, help='Number of images to generate. Overrides config value.')
    args = parser.parse_args()
    main(args)
