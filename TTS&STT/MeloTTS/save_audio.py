from melo.api import TTS as mTTS
from io import BytesIO
import torch
from getenv import GetEnv
from huggingface_hub import snapshot_download
from typing import Optional
import os
import pygame
import pdb

env = GetEnv()

def default_model():
    env.download_default_model()

def save_audio_file(text, language="KR", speed=1.2, output_name = "output.wav"):
    model_path = env.get_models_dir
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    env.download_default_model()
    ckpt, config = env.get_default_meloTTS_ckpt_and_config_path

    model = mTTS(language, device=device, config_path=config, ckpt_path=ckpt)
    speaker_ids = model.hps.data.spk2id

    output_path = os.path.join(env.get_output_dir, output_name)

    model.tts_to_file(text, speaker_ids[language], output_path=output_path , speed=1.2, format='wav')

if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(device)

    text = "온라인 구매와 차이점은 포낙 보청기를 사용할 때 많은 분들이 궁금해합니다. 사용자의 청력 상태나 환경, 사용 목적에 따라 온라인 구매와 차이점에 대한 적절한 정보가 만족도에 큰 영향을 줄 수 있습니다."

    save_audio_file(text)