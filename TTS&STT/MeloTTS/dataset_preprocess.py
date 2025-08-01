import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
project_melo = os.path.abspath(os.path.join(os.path.dirname(__file__), 'melo'))

if project_root not in sys.path:
    sys.path.append(project_root)
    sys.path.append(project_melo)

import re
import numpy as np
from typing import Optional

# libraries
from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict
import soundfile as sf
from tqdm import tqdm

# local
from getenv import GetEnv

env = GetEnv()

def clean_transcription(dataset : DatasetDict, key : str = "transcription"):
    def process_example(example):
            # Clean the transcription for a single example
            cleaned = re.sub(r"[^a-zA-Z0-9가-힣\s.,…]", "", example[key])
            return {key: cleaned}
    
    # Apply the cleaning function to each example in the 'train' split
    dataset['train'] = dataset['train'].map(process_example)
    return dataset

def clean_transcription_without_train(dataset, key : str = "transcription"):
    def process_example(example):
            # Clean the transcription for a single example
            cleaned = re.sub(r"[^a-zA-Z0-9가-힣\s.,…]", "", example[key])
            return {key: cleaned}
    
    # Apply the cleaning function to each example in the 'train' split
    dataset = dataset.map(process_example)
    return dataset
def load_dataset_only(dataset_id: str, apply_re : bool = True, **kwargs) -> DatasetDict:
    cache_dir = env.get_dataset_dir
    dataset =  load_dataset(dataset_id, cache_dir=cache_dir)
    if apply_re:
        dataset = clean_transcription(dataset, **kwargs)
    
    return dataset

def prepare_dataset(
    dataset_id: str,
    verbose: bool = True,
    apply_re: bool = True,
    make_wav: bool = True,
    **kwargs
) -> DatasetDict:
    """
    Load a dataset from Hugging Face and perform optional processing.

    Args:
        dataset_id (str): The ID of the dataset to load.
        verbose (bool): Whether to print the dataset details.
        apply_re (bool): Whether to apply regex cleaning to the transcription.
        make_wav (bool): Whether to save audio files as .wav.
        **kwargs: Additional arguments passed to processing functions.

    Returns:
        DatasetDict: The loaded and processed dataset.
    """
    dataset_dir = env.get_dataset_dir
    dataset = load_dataset(dataset_id, cache_dir=dataset_dir)
    local_dataset_name = dataset_id.split("/")[1]

    if apply_re:
        # Clean transcriptions in the dataset using regex
        clean_transcription(dataset, **kwargs)
    if verbose:
        # Print dataset details
        print(dataset, **kwargs)

    # Check if the local directory for the dataset exists, create if not
    if not os.path.exists(os.path.join(dataset_dir, local_dataset_name)):
        os.mkdir(os.path.join(dataset_dir, local_dataset_name))

    if make_wav:
        # Save audio files as .wav in the local directory
        save_dir = os.path.join(dataset_dir, local_dataset_name)
        for i, data in tqdm(enumerate(dataset['train'])):
            audio = data['audio']
            if isinstance(audio, dict) and 'array' in audio and 'sampling_rate' in audio:
                audio_data = audio['array']
                sampling_rate = audio['sampling_rate']
                file_name = os.path.join(save_dir, f"korean_{i}.wav")
                sf.write(file_name, audio_data, sampling_rate)

    return dataset, save_dir

def make_genshin_nahida_korean_dataset(audio_dir : os.PathLike, dataset : DatasetDict, text_key : str = "transcription", speaker_name : str = "KR-default", language_code : str = "KR"):
    # 오디오 음성과 dataset의 text_key 값의 순서가 일치해야함
    audio_files = os.listdir(audio_dir)
    texts = dataset['train'][text_key]

    output_name = "metadata.list"
    output_path = os.path.join(audio_dir, output_name)

    # 기존 메타데이터가 존재하면 길이에서 안맞으므로 삭제
    if os.path.exists(output_path):
        os.remove(output_path)
    
    if len(audio_files) != len(texts):
        raise ValueError(f"The number of audio files : {len(audio_files)} does not match the number of text files : {len(texts)}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for audio_file, text in zip(audio_files, texts):
            audio_path = os.path.join(audio_dir, audio_file)
            line = f"{audio_path}|{speaker_name}|{language_code}|{text}"
            f.write(line + "\n")
    
    print(f"Please run the following command manually:\ncd {project_melo} \npython preprocess_text.py --metadata {output_path}")


def make_genshin_nahida_korean_meloTTS_dataset(repo_id, **kwargs):
    """
    Downloads a dataset, preprocesses it, and saves it as a local meloTTS dataset.

    Args:
        repo_id (str): The ID of the dataset in the Hugging Face Hub.
        **kwargs:
            text_key (str): The key in the dataset used for transcription. Default: "transcription"\n
            speaker_name (str): Speaker name to be used in metadata. Default: "KR-default"\n
            language_code (str): Language code to be used in metadata. Default: "KR"\n
            apply_re (bool): Whether to apply regex-based cleaning to text. Default: True\n
            make_wav (bool): Whether to save audio as .wav files. Default: True\n
            verbose (bool): Whether to print dataset structure and details. Default: True\n
    """
    dataset, save_dir = prepare_dataset(repo_id, **kwargs)
    make_genshin_nahida_korean_dataset(audio_dir = save_dir, dataset = dataset, **kwargs)



def make_simon3000_dataset(repo_id : Optional[str] = None):
    dataset_dir = env.get_dataset_dir
    save_dir = os.path.join(dataset_dir, 'starrail-voice')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 다국어 / 200GB
    if repo_id is None:
        repo_id = "simon3000/starrail-voice"

    dataset = load_dataset(repo_id, split='train', cache_dir=dataset_dir)

    # language 가 Korean, Transcription이 빈칸이 아닌 컬럼으로 필터링
    def filtering(subset):
        return subset['language'] == "Korean" and subset['transcription'].strip() != "" and subset['speaker'] != ""
    
    filtered_dataset = dataset.filter(filtering)
    # 특수문자 필터링
    filtered_dataset = clean_transcription_without_train(filtered_dataset)
    """
    Dataset({
        features: ['audio', 'ingame_filename', 'transcription', 'language', 'speaker', 'voice_type'],
        num_rows: 34035
    })    
    """
    
    # MeloTTS에 적합한 형태로 데이터셋 전처리 output은 train/dataset/{speaker_name}/...wav, metadata.list 형태
    metadata = {}
    def sanitize_filename(name: str) -> str:
        # Windows에서 사용할 수 없는 문자: \ / : * ? " < > |
        return re.sub(r'[\\/*?:"<>|]', "_", name)
    
    for i, subset in tqdm(enumerate(filtered_dataset), total=len(filtered_dataset)):
        speaker_raw = subset['speaker']
        speaker = sanitize_filename(speaker_raw)
        audio = subset['audio']
        transcription = subset['transcription'].strip()
        language_code = "KR"

        speaker_dir = os.path.join(save_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)

        # 파일이름
        file_name = f"Korean_{i}.wav"
        file_path = os.path.join(speaker_dir, file_name)
        # 오디오 저장
        sf.write(file_path, audio['array'], audio['sampling_rate'])

        # metadata 저장 준비
        if speaker not in metadata:
            metadata[speaker] = []
        
        metadata[speaker].append(f"{file_path}|{speaker}|{language_code}|{transcription}")

    # speaker 별 metadata.list 작성
    for speaker, lines in metadata.items():
        metadata_path = os.path.join(save_dir, speaker, 'metadata.list')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

def remove_english_from_metadata(save_dir : str):
    """
    metadata.list에서 오디오 파일에는 존재하지 않지만, transcription 으로 존재하는 영어 대소문자를 제거 (color, nickname 등)
    """
    metadata_filename = "metadata.list"
    specific_token_pattern = re.compile(r"colordbc291ff", re.IGNORECASE)  # 대소문자 무시
    english_pattern = re.compile(r"[a-zA-Z]")


    for speaker_name in os.listdir(save_dir):
        speaker_path = os.path.join(save_dir, speaker_name)
        if not os.path.isdir(speaker_path):
            continue

        metadata_path = os.path.join(speaker_path, metadata_filename)
        if not os.path.exists(metadata_path):
            continue

        cleaned_lines = []

        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) != 4:
                    continue

                audio_path, speaker, language_code, text = parts

                # Step 1: 특정 문자열 제거
                text = specific_token_pattern.sub(" ", text)

                # Step 2: 영어 알파벳 제거
                text = english_pattern.sub("", text)

                cleaned_line = f"{audio_path}|{speaker}|{language_code}|{text.strip()}"
                cleaned_lines.append(cleaned_line)

        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))
        
        print("DONE")

def replace_audio_path_in_metadata(base_path: str, old_path_prefix: str, new_path_prefix: str):
    """
    base_path 내 모든 metadata.list에서 old_path_prefix → new_path_prefix로 audio 경로를 치환
    """

    metadata_filename = "metadata.list"
    old_path_prefix = os.path.normpath(old_path_prefix)
    new_path_prefix = os.path.normpath(new_path_prefix)
    base_path = os.path.normpath(base_path)

    for speaker_name in os.listdir(base_path):
        speaker_dir = os.path.join(base_path, speaker_name)
        metadata_path = os.path.join(speaker_dir, metadata_filename)

        if not os.path.isdir(speaker_dir) or not os.path.exists(metadata_path):
            continue

        updated_lines = []

        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) != 4:
                    continue

                audio_path, speaker, language_code, text = parts

                # 문자열 치환
                new_audio_path = audio_path.replace(old_path_prefix, new_path_prefix)
                if '\\' in new_audio_path:
                    new_audio_path = new_audio_path.replace('\\', '/')
                updated_lines.append(f"{new_audio_path}|{speaker}|{language_code}|{text}")

        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(updated_lines))

    print(f"✅ '{base_path}' 내 metadata.list 경로 변경 완료: '{old_path_prefix}' → '{new_path_prefix}'")

def replace_audio_path_in_metadata_file(metadata_path : str, old_path_prefix : str, new_path_prefix : str):
    """
    지정한 metadata.list 파일에서 audio_path의 old_path_prefix를 new_path_prefix로 치환합니다.

    Parameters:
    - metadata_path: metadata.list 파일의 전체 경로
    - old_path_prefix: 변경 전 경로 prefix
    - new_path_prefix: 변경 후 경로 prefix
    """

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Did not found {metadata_path}")
    
    old_path_prefix = os.path.normpath(old_path_prefix)
    new_path_prefix = os.path.normpath(new_path_prefix)

    updated_lines = []

    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 4:
                continue

            audio_path, speaker, language_code, text = parts

            # 경로 치환
            new_audio_path = audio_path.replace(old_path_prefix, new_path_prefix)

            if "\\" in new_audio_path:
                new_audio_path = new_audio_path.replace("\\", "/")
            
            updated_lines.append(f"{new_audio_path}|{speaker}|{language_code}|{text}")

    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(updated_lines))
    
    print("DONE")




if __name__ == "__main__":
    # 첫번째 데이터셋, 자동
    # repo_id = "habapchan/genshin-nahida-korean"
    # make_genshin_nahida_korean_meloTTS_dataset(repo_id)

    # 수동, 1. 데이터셋 만들기 + 음성파일 만들기
    # dataset, _ = prepare_dataset()

    # 수동, 2. meloTTS 학습에 필요한 데이터로 전처리
    # dataset = load_dataset_only()
    # audio_dir = r"E:\st002\repo\generative\audio\MeloTTS\train\dataset\genshin-nahida-korean"
    # make_dataset(audio_dir=audio_dir, dataset=dataset)

    # 두번째 데이터셋, 자동
    # make_simon3000_dataset()
    
    # metadata.list의 transcription 에서 영어 대/소문자 제거
    path = r"D:\audio_train"
    remove_english_from_metadata(path)