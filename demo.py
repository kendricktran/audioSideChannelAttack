import pyaudio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import cv2
from torchvision import models
from keyLabel import keyLabel
from config import config
import scipy.signal
import warnings

# Load the trained model
model_weights = "resnet_model_weights.pth"
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, config.num_classes)
model.load_state_dict(torch.load(model_weights))
model.eval()

# Initialize label encoder
labeler = keyLabel()
labeler.fit(['q','w','e','r','t','y','u','i','o','p','backspace',
             'a','s','d','f','g','h','j','k','l',
             'z','x','c','v','b','n','m','space'])

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
CHUNK = 1024

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Listening...")

def bandpass_filter(audio_data, lowcut=1000, highcut=5000, fs=44100, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    y = scipy.signal.lfilter(b, a, audio_data)
    return y

def is_silent(audio_data, threshold_db=-50):
    # Convert to float and normalize
    audio_data = audio_data.astype(np.float32) / 32768.0
    # Apply bandpass filter
    filtered_data = bandpass_filter(audio_data)
    # Calculate RMS in decibels
    rms = np.sqrt(np.mean(filtered_data**2))
    dB = 20 * np.log10(rms + 1e-10)  # Avoid log(0)
    return dB < threshold_db

def isolate_peaks(audio_data, sr):
    # Convert to float and normalize
    audio_data = audio_data.astype(np.float32) / 32768.0
    
    # Detect onsets using librosa
    onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sr, hop_length=128)
    peaks = librosa.util.peak_pick(
        onset_strength,
        pre_max=10,
        post_max=10,
        pre_avg=15,
        post_avg=15,
        delta=.5,
        wait=60
    )
    
    # Convert peaks to time domain
    if len(peaks) > 0:
        peak_sample = librosa.frames_to_samples(peaks[0], hop_length=128)
        start = max(0, peak_sample - 1024)
        end = min(len(audio_data), peak_sample + 1024)
        return audio_data[start:end]
    return None

try:
    buffer = np.array([], dtype=np.float32)
    while True:
        # Read audio data from the stream
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Add to buffer
        buffer = np.concatenate([buffer, audio_data.astype(np.float32) / 32768.0])
        
        # Process when we have enough data (e.g., 1 second)
        if len(buffer) >= RATE:  # 1 second at 22.05kHz
            # Find peaks in the buffer
            segment = isolate_peaks(buffer, RATE)
            
            if segment is not None and len(segment) >= 1024:
                # Process audio to mel spectrogram
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mel_spectrogram = librosa.feature.melspectrogram(
                        y=segment,
                        sr=RATE,
                        hop_length=16
                    )
                
                mel_spectrogram = cv2.resize(mel_spectrogram, (224, 224))
                mel_spectrogram = np.stack([mel_spectrogram] * 3, axis=0)

                # Convert to tensor and predict
                x = torch.from_numpy(mel_spectrogram).unsqueeze(0)
                logits = model(x.float())
                probabilities = F.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                predicted_class = labeler.inverse_transform([predicted_class.item()])[0]
                print(f"Predicted Key: {predicted_class}")
            
            # Reset buffer
            buffer = np.array([], dtype=np.float32)

except KeyboardInterrupt:
    print("Stopped listening.")

finally:
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

# p = pyaudio.PyAudio()
# for i in range(p.get_device_count()):
#     info = p.get_device_info_by_index(i)
#     print(f"Device {i}: {info['name']}")
