import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
project_melo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'melo'))

if project_root not in sys.path:
    sys.path.append(project_root)
    sys.path.append(project_melo)

import re

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

def make_dataset(audio_dir : os.PathLike, dataset : DatasetDict, text_key : str = "transcription", speaker_name : str = "KR-default", language_code : str = "KR"):
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
    
    print(f"Please run the following command manually:\ncd {project_melo} \npython preprocess_text.py --metadata {output_path}", 'green')


def make_meloTTS_dataset(repo_id, **kwargs):
    dataset, save_dir = prepare_dataset(repo_id, **kwargs)
    make_dataset(audio_dir = save_dir, dataset = dataset, **kwargs)







if __name__ == "__main__":
    repo_id = "habapchan/genshin-nahida-korean"
    # 데이터셋 만들기 + 음성파일 만들기
    # dataset, _ = prepare_dataset()

    # meloTTS 학습에 필요한 데이터로 전처리
    # dataset = load_dataset_only()
    # audio_dir = r"E:\st002\repo\generative\audio\MeloTTS\train\dataset\genshin-nahida-korean"
    # make_dataset(audio_dir=audio_dir, dataset=dataset)
