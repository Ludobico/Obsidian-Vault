"""
melotts에서 그냥 학습 돌리면 자동으로 설치되는 모델이 있는데 그걸로는 잘 안됩니다
MeloTTS안에 melo폴더에 download_utils.py 열어보면
다운로드 DOWNLOAD_CKPT_URL가 있는데
CKPT받으시면 여기서 KR에 있는거 받으시고 그걸 G_0.pth로 이름을 바꾸시고
밑에 PRETRAINED_MODELS에서 D.pth랑 DUR.pth받으신 후에
학습폴더 안에 넣으시면 한국어 학습이 됩니다
"""
# 위 의견을 반영하여 download_utils.py 에서 필요한 부분만 가져와서 전처리

import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
project_melo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'melo'))

if project_root not in sys.path:
    sys.path.append(project_root)
    sys.path.append(project_melo)

from getenv import GetEnv

# libraries
from pathlib import Path
from filelock import FileLock
from urllib.request import urlretrieve
import requests
from tqdm import tqdm


DOWNLOAD_CKPT_URLS = {
    'KR': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/KR/checkpoint.pth',
}

PRETRAINED_MODELS = {
    'D.pth': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/pretrained/D.pth',
    'DUR.pth': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/pretrained/DUR.pth',
}

LANG_TO_HF_REPO_ID = {
    'EN': 'myshell-ai/MeloTTS-English',
    'EN_V2': 'myshell-ai/MeloTTS-English-v2',
    'EN_NEWEST': 'myshell-ai/MeloTTS-English-v3',
    'FR': 'myshell-ai/MeloTTS-French',
    'JP': 'myshell-ai/MeloTTS-Japanese',
    'ES': 'myshell-ai/MeloTTS-Spanish',
    'ZH': 'myshell-ai/MeloTTS-Chinese',
    'KR': 'myshell-ai/MeloTTS-Korean',
}

# def download_file(url, dst_path):
#     dst_path = Path(dst_path)
#     dst_path.parent.mkdir(parents=True, exist_ok=True)

#     if dst_path.exists():
#         print(f"{dst_path.name} Alreadt exists. Skipping download")
#         return
    
#     with FileLock(str(dst_path) + ".lock"): # 동시 다운로드 방지
#         urlretrieve(url, dst_path)

def download_file(url, dst_path):
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists():
        print(f"{dst_path.name} already exists. Skipping download")
        return

    lock_path = str(dst_path) + ".lock"
    with FileLock(lock_path):  # 동시 다운로드 방지
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1024

        progress_bar = tqdm(
            total=total_size, unit='B', unit_scale=True, desc=dst_path.name
        )

        with open(dst_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        progress_bar.close()

def check_training_files(target_dir, folder_name : str = "train"):
    target_dir = Path(target_dir)

    # 확인할 파일 목록
    check_files = [
        "config.json", "D.pth", "DUR.pth", "G_0.pth",
        "metadata.list", "metadata.list.cleaned",
        "train.list", "val.list"
    ]

    print("\n[🔍] Checking required training files in:", target_dir)
    for filename in check_files:
        file_path = target_dir / filename
        if file_path.exists():
            print(f"  ✓ {filename} exists")
        else:
            print(f"  ✗ {filename} missing")

    
def prepare_pretrained_models(target_dir):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    kr_ckpt_url = DOWNLOAD_CKPT_URLS['KR']
    g_model_path = target_dir / "G_0.pth"
    download_file(kr_ckpt_url, g_model_path)

    for name, url in PRETRAINED_MODELS.items():
        download_file(url, target_dir / name)
    
    check_training_files(target_dir)

    print("DONE")

if __name__ == "__main__":
    target_dir = r"E:\st002\repo\generative\audio\MeloTTS\train\dataset\genshin-nahida-korean"
    prepare_pretrained_models(target_dir)