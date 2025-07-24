```python

device = "cuda" if torch.cuda.is_available() else 'cpu'

env = GetEnv()

def save_audio_file(text, G_model_path, config_path, language="KR", speed=1.0, output_name = "output.wav"):
    """
    G_model_path : /path/to/checkpoint/G_<iter>.pth
    """
    model = mTTS(language, device=device, config_path=config_path, ckpt_path=G_model_path)
    # config의 "spk2id": {"str": int} 중 int에 해당하는 숫자
    speaker_id = 0

    output_path = os.path.join(env.get_output_dir, output_name)

    model.tts_to_file(text, speaker_id, output_path=output_path, speed=speed, format='wav')
```

