# Fine-Tune a Transformer-model-for-Language-Transcription

# Aim –
The aim of this lab is to perform Swedish text transcription using transformers. A pre-trained transformer model, Whisper has been used which is further fine-tuned on the Common voice dataset to achieve a higher performance. A serverless Gradio UI is further built on Hugging Face which takes the input from the model in the form of speech and provides a text output. 
The lab work is divided into the following pipelines – 
Feature engineering pipeline 
Training pipeline 
Inference pipeline 

# Requirements -
1. Common voice dataset 
2. OpenAI Whisper model
3. Colab with GPU 
4. Hugging Face repository 
5. Hugging Face Spaces 

# Dataset – 
The common voice dataset has been used for fine tuning. This dataset consists of 16413 validated hours in 100 languages. For this lab, the Swedish language has been used. 

# Steps –

# data_preparation.ipynb – 
The limited GPU provided by Google Colab has been used. The common voice dataset is downloaded from the hugging face website. Irrelevant columns such as accent, age, client_id, down_votes, gender, locale, path, segments and up_votes were removed. 
The audio is down sampled to 16Hz. This is followed by loading the Whisper feature extractor, the tokenizer, and the Whisper processor. 
The save_to_disk function stores the dataset on google drive as a tar.gz file which can further be used for in the training pipeline. 

# model_training.ipynb –
The dataset is loaded by untaring the tar.gz file in drive. A data collector is then defined which converts the input features into batched pytorch tensors and performs padding on the labels. The word error rate metric is loaded which will be used to evaluate our ASR system. 

Three models were trained –

1.The first model had a per device batch size of 16, learning rate of 1e-5 and 1 gradient accumulation step. It was trained on 4000 steps but due to the computational limitations, 2000 steps were completed. A checkpoint was created every 1000 steps. The training loss and validation loss were 0.0477 and 0.29057 respectively. The word error rate was 20.195746%. This is shown in model_training2.ipynb. The losses and the WER per checkpoint are given below -  

| Steps   | Training loss | Validation loss | WER.  |
| :------:| :-----------: | :-------------: | :---: | 
| 500     | 0.3103        |        0.326575 | 23.87%|  
| 1000    | 0.1383        |        0.295559 | 21.49%|
| 1500    | 0.1325        |        0.281871 | 20.84%|
| 2000    | 0.0477        |        0.290570 | 20.19%|

2.The second model had a per device batch size of 8 and the gradient accumulation step was increased to two. This model was run for 3000 steps. The loss and WER at each checkpoint is given below –

| Steps   | Training loss | Validation loss | WER.  |
| :------:| :-----------: | :-------------: | :---: | 
| 1000    | 0.139         |        0.295936 | 21.53%|  
| 2000    | 0.047         |        0.290317 | 20.35%|
| 3000    | 0.016         |        0.296569 | 19.70%|

3.We performed further hyperparameter tuning by experimenting with the learning rate. The learning rate was increased to 1e-3 by keeping all the other parameters the same. The resulted in a high increase in the WER and the losses. This model was run for 2000 steps with checkpoints created every 500 steps. At the 2000th step, the training and validation losses were 2.8829 and 3.135914 respectively while the WER was 122.25. This is shown in model_training3.ipynb. The loss and WER at each checkpoint is given below –

| Steps   | Training loss | Validation loss | WER.   |
| :------:| :-----------: | :-------------: | :---:  | 
| 500     | 4.4594        |        4.065455	| 132.38%|  
| 1000    | 3.3984        |        3.490493 | 106.75%| 
| 1500    | 3.1814        |        3.251461 | 102.50%|
| 2000    | 2.8829        |        3.135914 | 122.25%|

Clearly, the second model had a better performance, with a WER of 19.70% after 3000 steps. This model is uploaded to Hugging face. -
https://huggingface.co/Vasi001/whisper-small

We took a model centric approach by changing the per device batch size and the learning rate. Due to time constraints and less computational resources we could not train other models with different hyperparameters.  

# User Interface 

The hugging face UI link is given below. This allows the user to record their voice in Swedish and the model will transcribe it to text. -
https://huggingface.co/spaces/Vasi001/whisper-small
