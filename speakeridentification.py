from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import librosa
import numpy as np
import joblib

def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=_, n_mfcc=13)
    return np.mean(mfccs, axis=1)

def prepare_data(main_folder):
    features = []
    labels = []
    for speaker_folder in os.listdir(main_folder):
        speaker_path = os.path.join(main_folder, speaker_folder)
        if os.path.isdir(speaker_path):
            for filename in os.listdir(speaker_path):
                if filename.endswith(".wav"):
                    file_path = os.path.join(speaker_path, filename)
                    feature = extract_features(file_path)
                    labels.append(speaker_folder)  # Use the folder name as the label
                    features.append(feature)
    return np.array(features), np.array(labels)

# Replace the following path with the location of your dataset on your computer
main_folder_path = 'D:\\IIITH\\Dataset'

X, y = prepare_data(main_folder_path)

# Check if there are samples in your dataset
if len(X) == 0:
    print("No samples found in the dataset.")
else:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check if there are samples in the training set
    if len(X_train) == 0:
        print("No samples in the training set.")
    else:
        # Train a Support Vector Machine (SVM) classifier
        model = SVC()
        model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, 'speaker_model.joblib')

        # Validate the model
        y_test_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Test Accuracy: {accuracy}")
