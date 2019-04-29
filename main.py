# Khoa Nguyen; 272580; khoa.nguyen@tuni.fi
# Hung Nguyen; 272575; hung.nguyen@tuni.fi
# Lab 2 - Computational Auditory Scene Recognition: 
# Classification of Environmental Sounds with Python

#%% Importing libaries
import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
import os
import sklearn
import scipy
import operator
import statistics

#%% Assignment 1: Feature extraction

path = 'AudioData/signal.wav'
signal, fs = sf.read(path)
def framewise_feature_extraction(data, fs=8000):
    # input: the audio data and its sampling rate
    # output: The sub-band energy ratio for each frame in the audio data
    # which describes the relative energy on certain frequency bands 
    # for example: put in the signal.wav, we get the output as the 
    # subband_energy_ratios list with 62 lists of size 4x1
    
    # Load the signal    
    # Sampling frequency (sample rate/fs) of the signal.wav: 8000Hz (8000 samples per sec)
    duration = data.size / fs
    # Duration of the audio file: 0.937625s

    # Divide the audio clip into frames by using Hanning window

    frame_length = 0.03
    frame_size = int(fs * frame_length)
    # one frame contains 240 samples with fs = 8000

    overlap_length = 0.015
    overlapped_size = int(overlap_length / frame_length * frame_size) # number of overlapped samples
    # 120 samples overlapped

    frames = []
    start = 0
    while start < data.size:
        frames.append(data[start:(start+frame_size)])
        start = start + frame_size - overlapped_size
    # remove the last frame (because of repetition)
    frames = frames[0:len(frames)-1]

    # apply the window on the frames
    windowed_frames = []
    for i in range(len(frames)):
        windowed_frames.append(frames[i]*np.hanning(len(frames[i])))
    
    # plot the 50th windowed_frames
#    plt.figure()
#    plt.plot(windowed_frames[49])

    # For each signal frame, we will extract a feature vector x=[x(1),x(2),x(3),x(4)]T 
    # (The sub-band energy ratio) containing the relative energy on the frequency bands 
    # 0 – 0.5 kHz, 0.5 – 1 kHz, 1 – 2 kHz and 2 – 4 kHz.

    # To do so, first we need the DFT of each signal frame, 1024 bins
    N = 1024
    dft_frames = [np.fft.fft(i, n=N) for i in windowed_frames]
    
    # Plot the amplitude of the signal frame
#    plt.figure()
#    plt.plot(np.abs(dft_frames[49]))

    # Calculate the sub-band energy ratio
    # The frequency at bin FFT[k] is:
    # k * SamplingFrequency / N, if k <= [N/2] 
    # (N-k) * SamplingFrequency / N, if k >= [N/2] 
    # => k (index) = frequency * N / SamplingFrequency
    subband_energy_ratios = []
    # each signal frame has 4 frequency bands
    for i in range(len(dft_frames)):
        denominator = np.sum(np.abs(dft_frames[i][0:int(N/2)])**2)
        # 0 – 500 Hz frequency band
        first_band = dft_frames[i][0 : int(N*500/fs)]
        x1 = np.sum(np.abs(first_band)**2) / denominator 
        # 500 - 1000 Hz 
        second_band = dft_frames[i][int(N*500/fs) : int(N*1000/fs)]
        x2 = np.sum(np.abs(second_band)**2) / denominator
        # 1000 - 2000 Hz
        # What are the indices of the DFT bins belonging to the frequency band 1–2 kHz?
        # it is in the range 128:256
        third_band = dft_frames[i][int(N*1000/fs) : int(N*2000/fs)]
        x3 = np.sum(np.abs(third_band)**2) / denominator
        # 2000 - 4000 Hz
        fourth_band = dft_frames[i][int(N*2000/fs) : int(N*4000/fs)]
        x4 = np.sum(np.abs(fourth_band)**2) / denominator
    
        subband_energy_ratios.append([x1, x2, x3, x4]) 

#    plt.figure()
#    plt.bar(['x1', 'x2', 'x3', 'x4'], subband_energy_ratios[49])
    
    return subband_energy_ratios


#%% Assignment 2: Recognition of the recording environment

# 3.1 Data Preprocessing
audio_files_path = 'AudioData/UEA'


def data_preprocessing(audio_files_path):
    audio_names = os.listdir(path=audio_files_path)[1:]
    audio_names.sort()
    X_temp = []
    y = []
    for counter, value in enumerate(audio_names):
        # read each audio data
        data, fs = sf.read(audio_files_path + '/' + value)
        # segment data into 1 sec clips
        # fs = samples / sec => 1 sec has fs samples
        if data.size > 2400000:
            segments = [data[i:i + fs] for i in range(0, len(data), fs)][:-1]
        else:
            segments = [data[i:i + fs] for i in range(0, len(data), fs)]
        X_temp.append(segments)
        y.append(counter+1)
    
    X = []
    for i in range(len(X_temp)):
        for j in X_temp[i]:
            X.append(j)
            
    y = np.repeat(y, len(X)/len(audio_names))
    return X, y

X, y = data_preprocessing(audio_files_path)

#%% 3.2 Feature extraction

def feature_extraction(data):
    # apply the framewise_feature_extraction function on 
    # each sample, then compute the average of each subband energy value 
    # over the frames of the audio clip.
    # input: the audio signal
    # output: one average feature vector of size 4x1 for each audio clip.
    for i in X:
        subband_energy_ratios = framewise_feature_extraction(i)
        X_featured.append(np.average(subband_energy_ratios, axis=0))
    return X_featured


#%% Divide the clips among training and test data sets
X_featured = []
X_featured = feature_extraction(X)

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_featured, y,
                                                    test_size=0.2)


#%% 3.3 Classification

def most_common(l):
    # use to find the most common element in a list
    try:
        return statistics.mode(l)
    except statistics.StatisticsError as e:
        # will only return the first element if no unique mode found
        if 'no unique mode' in e.args[0]:
            return l[0]
        # this is for "StatisticsError: no mode for empty data"
        # after calling mode([])
        raise
#%%
def knn(training_samples, training_labels, new_data, k=1):
    # input: the set training samples (training feature vectors) with correct 
    # labels, a new test sample (test feature vector)
    # output: the predicted class of the new test sample
    
    # To classify a test vector new_data not included in the training data,
    # we go through all the training vectors in order to find the one that 
    # is closest to x, i.e. to find its nearest neighbor x′
    distance = []
    for i in training_samples:
        distance.append(np.linalg.norm(new_data - i))   
    minimum_distance_indices = np.argsort(distance)[0:k]
    minimum_distance_classes = []
    for i in minimum_distance_indices:
        minimum_distance_classes.append(training_labels[i])
    return most_common(minimum_distance_classes)

#%% training phase (1NN)
y_pred_1nn = []
for i in X_test:
    y_pred_1nn.append(knn(X_train, y_train, i))
    
print(sklearn.metrics.confusion_matrix(y_test, y_pred_1nn))
print(sklearn.metrics.accuracy_score(y_test, y_pred_1nn))

#%% training phase (5NN)
y_pred_5nn = []
for i in X_test:
    y_pred_5nn.append(knn(X_train, y_train, i, 5))
    
print(sklearn.metrics.confusion_matrix(y_test, y_pred_5nn))
print(sklearn.metrics.accuracy_score(y_test, y_pred_5nn))
    
    
#%% training phase (5NN)
y_pred_10nn = []
for i in X_test:
    y_pred_10nn.append(knn(X_train, y_train, i, 10))
    
print(sklearn.metrics.confusion_matrix(y_test, y_pred_10nn))
print(sklearn.metrics.accuracy_score(y_test, y_pred_10nn))    
    
