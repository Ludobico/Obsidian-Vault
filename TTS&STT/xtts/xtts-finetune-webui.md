
https://github.com/daswer123/xtts-finetune-webui

## Installation

you have to install [[Pytorch/CUDA/CUDA|CUDA]]

1. `git clone https://github.com/daswer123/xtts-finetune-webui`
2.  `cd xtts-finetune-webui`
3.  `pip install torch`
4.  `pip install -r requirements.txt`
5.  `python xtts_demo.py`

### If you're using Windows

**if you are using  an Anaconda virtual environment follow the Installation section and run `xtts_demo.py` **

1. First start `install.bat`
2. To start the server start `start.bat`
3. Go to the local address `127.0.0.1:5003`

### On Linux

**if you are using  an Anaconda virtual environment follow the Installation section and run `xtts_demo.py` **

1. Run `bash install.sh`
2. To start the server start `start.sh`
3. Go to the local address `127.0.0.1:5003`


## Fine tuning a custom voice

### Data processing

![[xtts1 1.png|1200]]

When you run a `xtts_demo.py` , you will see a screen like this, you need to prepare a dataset for fine tuning voice, this section (Data processing) is gonna help you to create trainable data


#### 01 Output path

A directory where your processed dataset will be stored, it is set to `finetune_models` by default, if this directory does not exist, it will be created automatically

#### 02 Audio file

select a wav or mp3 file in this section, if your source is youtube URL, you should convert it to mp3

#### 03 Whisper Model

Select a Whisper model debeloped by OpenAI. (It actually uses a `Whisper Faster model`)
Large models have more parametes, meaning higher accuracy, but they also require significantly more VRAM and processing time

**Supported models**

- small
- medium
- large
- large-v2
- large-v3

#### 04 Dataset Language

Select the language of your source


![[Pasted image 20250123105811.png]]

When `data processing` is complete. your directory will contain a folder where the processed data is stored.

`lang.txt` : the language you selected

`wavs` : the chunked audio data from your source by Whisper model

`metadata_eval.csv` : Metadata used to evaluate scripts from wavs

`metadata_train.csv` : Metadata used to train scripts from wavs


### Fine-tuning XTTS Encoder

![[123123.jpg]]

#### 01 Load Params from output folder

Literally

#### 02 XTTS base version

basically, XTTS development stopped in 2024 , I assume version `v2.0.3` might be the last one

#### 03 train, eval CSV

paths to metadata for training and evaluation.

#### 04 Hyperparameter

I recommend referring to [[XTTS Model Finetuning Guide (Advanced Version)]]


#### Training and optimization

![[Pasted image 20250123133637.png]]
you will see these buttons at the bottom.
Click the `Run the training` button for training model using your processed data.
Once the training is done, click the `Optimize the model` for optimization your pretrained model.


