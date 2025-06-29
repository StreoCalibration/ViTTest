import os
import glob
import shutil

def clear_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"폴더가 존재하지 않습니다: {folder_path}")
        return
    for file_path in glob.glob(os.path.join(folder_path, '**', '*'), recursive=True):
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"삭제됨: {file_path}")
            except Exception as e:
                print(f"삭제 실패: {file_path} - {e}")
        elif os.path.isdir(file_path):
            try:
                shutil.rmtree(file_path)
                print(f"폴더 삭제됨: {file_path}")
            except Exception as e:
                print(f"폴더 삭제 실패: {file_path} - {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    base_dir = os.path.join(script_dir, "dataset_root", "synthetic_data")
    clear_folder(base_dir)