# import packages
import numpy as np
import pandas as pd
import librosa as lr
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from numpy.lib.stride_tricks import sliding_window_view
import tensorflow as tf


def wav_to_ts(audio_wav, sr=22050):
    """
    Function to obtain the time series representation of a signal from a .wav file
    audio_wav: file system path to access .wav file
    sr: sampling rate

    """
    audio_ts, sr = lr.load(audio_wav, sr)
    return audio_ts


def extract_features(audio_ts, sr=22050):
    """
    Function to extract features from the time series representation of a signal
    audio_ts: time waveform expressed as a time series
    sr: sampling rate (Hz)
    """
    # spectral centroid
    spec_cent = lr.feature.spectral_centroid(y=audio_ts)[0]

    # spectral rolloff
    spec_rolloff = lr.feature.spectral_rolloff(y=audio_ts)[0]

    # spectral bandwidth (p=2)
    spec_band_2 = lr.feature.spectral_bandwidth(y=audio_ts, p=2)[0]

    # spectral bandwidth (p=3)
    spec_band_3 = lr.feature.spectral_bandwidth(y=audio_ts, p=3)[0]

    # tempo - rythmic feature
    tempo = lr.beat.tempo(y=audio_ts, sr=sr, aggregate=None)

    # mel cepstral coefficient
    num_mcc_coeff = 5
    mcc = lr.feature.mfcc(y=audio_ts, n_mfcc=num_mcc_coeff)

    # zero crossing rate
    zero_cross_rate = lr.feature.zero_crossing_rate(y=audio_ts)[0]

    # caculate tempo
    tempo = lr.beat.tempo(y=audio_ts, sr=sr, aggregate=None)

    feature_dict = {'spectral_centroid': spec_cent, 'spectral_rolloff': spec_rolloff,
                         'spectral_bandwidth_2': spec_band_2, 'spectral_bandwith_3': spec_band_3,
                    'zero_crossing_rate': zero_cross_rate, 'tempo': tempo}
    mccf_dict = {'mcc_' + str(i+1): mcc[i] for i in list(range(num_mcc_coeff))}
    feature_dict.update(mccf_dict)
    return pd.DataFrame(feature_dict)


def texturize_features(audio_ts, sr=22050, texture_window_length=1000):
    """
    Function to aggregate extracted features over a texture window (larger than analysis window)
    audio_ts: time waveform expressed as a time series
    sr: sampling rate
    texture_window_length: length of texture window for feature aggregation (in ms)
    """

    analysis_window = (np.power(10, 6)) / (sr)
    num_frames = int(texture_window_length / analysis_window)
    features = extract_features(audio_ts)
    features_rolling = features.rolling(window=num_frames)
    agg_features = pd.concat([features_rolling.apply(np.mean), features_rolling.apply(np.var)], axis=1)
    agg_features.columns = [i + '_rolling_mean' for i in features.columns] + [i + '_rolling_var' for i in
                                                                              features.columns]
    return agg_features


def extract_and_texturize_batch(audio_wav_list, sr=22050, texture_window_length=1000):
    """
    Function to batch process a list of .wav files and return features aggregated over texture window

    audio_wav_list: list/array of file system path strings to .wav files of audio
    sr: sampling rate (Hz)
    texture_window_length: length of texture window for feature aggregation (in ms)
    """

    df = pd.DataFrame()
    for wav_file in audio_wav_list:
        audio_ts, sr = wav_to_ts(wav_file, sr)
        df_temp = texturize_features(audio_ts, sr, texture_window_length)
        df_temp['audio_file'] = wav_file
        df_temp['audio_index'] = df_temp.index
        df = pd.concat([df, df_temp])
    return df


def train_baseline_model(audio_wav_list, method='LOF', sr=22050, texture_window_length=1000):
    """
    Function to batch process a list of .wav files and return features aggregated over texture window

    audio_wav_list: list/array of file system path strings to .wav files of audio
    method: method used for anomaly detection - either Local Outlier Factor/ Isolation Forest
    sr: sampling rate (Hz)
    texture_window_length: length of texture window for feature aggregation (in ms)
    """

    methods = ['LOF', 'IsoForest']
    if method not in methods:
        raise ValueError("chosen method must be one of %m." % methods)
    #extract features and aggregate over texture window
    df_features = extract_and_texturize_batch(audio_wav_list, sr, texture_window_length)
    feature_cols = df_features.columns

    # scale features
    scaler = MinMaxScaler()
    df_train = pd.DataFrame(scaler.fit_transform(df_features), columns=feature_cols, index=df_features.index)

    # train model
    if method == 'LOF':
        model = LocalOutlierFactor()
        model_fitted = model.fit(df_train)
    elif method == 'IsoForest':
        model = IsolationForest()
        model_fitted = model.fit(df_train)
    return scaler.fit(df_features), model_fitted


def extract_features_autoencoder(audio_ts, sr=22050, n_mels=128, n_frames=5):
    """
    Function to extract mel spectogram based features for training an autoencoder

    audio_ts: time waveform expressed as a time series
    sr: sampling rate

    """
    mel_spec = lr.feature.melspectrogram(y=audio_ts, sr=sr, n_mels=n_mels)
    mel_spec = np.transpose(mel_spec)
    df = pd.DataFrame()
    for i in np.arange(n_frames-1, -1, -1):
        #print(i)
        df= pd.concat([df, pd.DataFrame(mel_spec).shift(i)], axis=1)

    df.columns = ['feature_' + str(i+1) for i in range(n_frames*n_mels)]
    return df.dropna()


def train_autoencoder_model(X_train, n_layers= [128, 128, 128, 128, 32, 128, 128, 128, 128]):
    X_train_transformed = MinMaxScaler().fit(X_train)
    # data dimensions // hyperparameters
    input_dim = X_train_transformed.shape[1]
    BATCH_SIZE = 256
    EPOCHS = 100

    # https://keras.io/layers/core/
    autoencoder = tf.keras.models.Sequential([

        # deconstruct / encode
        tf.keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dense(32, activation='elu'),

        # reconstruction / decode
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dense(input_dim, activation='elu')

    ])

    # https://keras.io/api/models/model_training_apis/
    autoencoder.compile(optimizer="adam",
                        loss="mse",
                        metrics=["acc"])

    # print an overview of our model
    autoencoder.summary();