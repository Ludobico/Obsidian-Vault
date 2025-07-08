import os, sys
from typing import Optional, Union
from huggingface_hub import snapshot_download

class GetEnv:
    def __init__(self, base_dir : Optional[Union[os.PathLike, str]] = None):
        self.curdir = os.path.abspath(base_dir) if base_dir else os.path.dirname(os.path.abspath(__file__))
        self._ensure_dirs_exist()

    def _ensure_dirs_exist(self):
        os.makedirs(self.get_train_dir, exist_ok=True)
        os.makedirs(self.get_dataset_dir, exist_ok=True)
        os.makedirs(self.get_models_dir, exist_ok=True)
        os.makedirs(self.get_output_dir, exist_ok=True)

    @property
    def get_train_dir(self):
        train_dir = os.path.join(self.curdir, 'train')
        return train_dir
    @property
    def get_dataset_dir(self):
        train_dir = self.get_train_dir
        dataset_dir = os.path.join(train_dir, 'dataset')
        return dataset_dir
    @property
    def get_models_dir(self):
        train_dir = self.get_train_dir
        models_dir = os.path.join(train_dir, 'models')
        return models_dir
    
    @property
    def get_output_dir(self):
        train_dir = self.get_train_dir
        output_dir = os.path.join(train_dir, 'output')
        return output_dir

    @property
    def get_default_meloTTS_repo_id(self):
        default_meloTTS_repo_id = "myshell-ai/MeloTTS-Korean"
        return default_meloTTS_repo_id
    
    def download_default_model(self):
        models_dir = self.get_models_dir
        repo_id = self.get_default_meloTTS_repo_id
        
        if not os.path.exists(os.path.join(models_dir, repo_id)):
            snapshot_download(repo_id=repo_id, local_dir=self.get_default_meloTTS_model_dir, local_dir_use_symlinks=False)
    
    @property
    def get_default_meloTTS_model_dir(self):
        """
        Call `download_default_model()` before using this method.

        This property is used to download the default Korean MeloTTS model
        if it is not already downloaded. The downloaded model is stored
        in the 'default_model' directory within the 'models' directory.

        Returns:
            str: The path to the 'default_model' directory.
        """
        models_dir = self.get_models_dir
        default_meloTTS_model_dir = os.path.join(models_dir, 'default_model')
        return default_meloTTS_model_dir
    
    @property
    def get_default_meloTTS_ckpt_and_config_path(self):
        """
        Call `download_default_model()` before using this method.

        Returns the paths to the default Korean MeloTTS model's checkpoint and config file.

        The default Korean MeloTTS model is downloaded by calling the `download_default_model()`
        method. The model is stored in the 'default_model' directory within the 'models'
        directory.

        Returns:
            tuple: A tuple containing the path to the 'checkpoint.pth' file and the path
                to the 'config.json' file of the default Korean MeloTTS model.
        """
        default_meloTTS_model_dir = self.get_default_meloTTS_model_dir
        ckpt = os.path.join(default_meloTTS_model_dir, 'checkpoint.pth')
        config = os.path.join(default_meloTTS_model_dir, 'config.json')
        return (ckpt, config)