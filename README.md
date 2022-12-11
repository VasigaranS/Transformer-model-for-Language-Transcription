# Fine-Tune a Transformer-model-for-Language-Transcription

#Aim –
The aim of this lab is to perform Swedish text transcription using transformers. A pre-trained transformer model, Whisper has been used which is further fine-tuned on the Common voice dataset to achieve a higher performance. A serverless Gradio UI is further built on Hugging Face which takes the input from the model in the form of speech and provides a text output. 
The lab work is divided into the following pipelines – 
Feature engineering pipeline 
Training pipeline 
Inference pipeline 

#Dataset – 
The common voice dataset has been used for fine tuning. This dataset consists of 16413 validated hours in 100 languages. For this lab, the Swedish language has been used. 

#Steps –
data_prep.ipynb – 
The limited GPU provided by Google Colab has been used. The common voice dataset is downloaded from the hugging face website. Irrelevant columns such as accent, age, client_id, down_votes, gender, locale, path, segments and up_votes were removed. 
The audio is down sampled to 16Hz. This is followed by loading the Whisper feature extractor, the tokenizer, and the Whisper processor. 
The save_to_disk function stores the dataset on google drive as a tar.gz file which can further be used for in the training pipeline. 
whisper_sv.ipynb –
The dataset is loaded by untaring the tar.gz file in drive. A data collector is then defined which converts the input features into batched pytorch tensors and performs padding on the labels. The word error rate metric is loaded which will be used to evaluate our ASR system. Two models were trained –
The first model had a batch size of 16, learning rate of 1e-5 and 1 gradient accumulation step. It was trained on 4000 steps but due to the computational limitations, 1673 steps were completed. A checkpoint was created at 1000 steps. The training loss and validation loss were 0.1383 and 0.29559 respectively. The word error rate was 21.49%
The second model had a batch size of 8. This model was run for 3000 steps. The loss and WER at each checkpoint is given below –


Clearly, the second model had a better performance, with a WER of 19.69% after 3000 steps.
