import joblib
import librosa
import numpy as np

def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=_, n_mfcc=13)
    return np.mean(mfccs, axis=1)

def predict_speaker(model, new_feature):
    # Reshape the feature to match the model's expectations
    new_feature = new_feature.reshape(1, -1)

    # Make a prediction
    predicted_label = model.predict(new_feature)
    return predicted_label[0]

# Load the trained model
model = joblib.load('D:\\IIITH\\speaker_model.joblib')

# Replace 'your_audio_file.wav' with the path to the audio file you want to predict
new_feature = extract_features('D:\\IIITH\\Marget.wav')

# Make a prediction
predicted_speaker = predict_speaker(model, new_feature)

# Print the predicted speaker
print(f"Predicted Speaker: {predicted_speaker}")
