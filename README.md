# Speaker-Recognition
Speaker identification using the MFCC algorithm 
This repository contains code for a speaker recognition project using the Mel-Frequency Cepstral Coefficients (MFCC) algorithm. The project utilizes the scikit-learn library for Support Vector Machine (SVM) classification
Project Structure
# speakeridentification.py:
This script is used for training the speaker recognition model. It extracts MFCC features from audio files in the dataset, prepares the data, trains an SVM classifier, and saves the trained model.

# answer.py: 
This script demonstrates how to use the trained model to predict the speaker of a new audio file. It loads the pre-trained model, extracts MFCC features from a new audio file, and predicts the speaker using the SVM model.

# Dataset
The dataset consists of audio samples from the following speakers:

Benjamin_Netanyau
Jens_Stoltenberg
Julia_Gillard
Magaret_Tarcher
Nelson_Mandela

You can get datset from https://www.kaggle.com/kongaevans/speaker-recognition-dataset


# USAGE 

# Install Dependencies:
pip install -r requirements.txt

# Training:
python speakeridentification.py

# Prediction:
python answer.py

# Note
Ensure that your dataset is organized with each speaker's audio samples in a separate folder.

Adjust the paths in the code according to your dataset and file locations.

The trained model (speaker_model.joblib) will be saved in the same directory as the training script.


Feel free to explore and modify the code for further improvements or customization.



