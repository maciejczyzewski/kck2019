# Usage:
# $ python3 bin/5.py --do --path='data/trainall/049_M.wav'
# $ sox -t alsa default recording.wav silence 1 0.1 5% 1 1.0 5%

################################################################################
# Used materials:
# https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
# https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
# https://github.com/jameslyons/python_speech_features
################################################################################

try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys

    sys.excepthook = IPython.core.ultratb.ColorTB()

import os
import time
import pickle
import librosa
import argparse
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 44100
DATASET_FILES = "data/trainall/*.wav"
MODEL_PATH = "treeboost_exp.txt"


def save_dataset(name, data):
    with open(f"data/{name}.pickle", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataset(name):
    with open(f"data/{name}.pickle", "rb") as handle:
        return pickle.load(handle)


def fbanks(signal, NFFT=512, nfilt=40):
    """Author: Haytham Fayek"""
    global SAMPLE_RATE
    mag_frames = np.absolute(np.fft.rfft(signal, NFFT))  # Magnitude of the FFT
    pow_frames = (1.0 / NFFT) * ((mag_frames)**2)  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(
        1 + (SAMPLE_RATE / 2) / 700)  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel,
                             nfilt + 2)  # Equally spaced in Mel scale
    hz_points = 700 * (10**(mel_points / 2595) - 1)  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / SAMPLE_RATE)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0,
                            np.finfo(float).eps,
                            filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)
    return filter_banks


class Dataset:
    def __init__(self, files=None, name="dataset"):
        self.files = files
        self.name = name
        self.load(self.name)

    def load(self, name="dataset"):
        if os.path.exists(f"data/{name}.pickle"):
            self.X, self.y = load_dataset(name)

    def sample(self, path, window):
        features = Dataset.features(window)
        print("[", features.shape, end="]", flush=True)
        if path.endswith("_K.wav"):
            label = 1
        else:
            label = 0
        return features, label

    def create(self):
        X, y = [], []

        for path in tqdm(files):
            print(f"path={path}")
            signal, sample_rate = Dataset.audio(path)
            windows = Dataset.windows(signal, sample_rate=sample_rate)

            for window in windows:
                features, label = self.sample(path, window)
                X.append(features)
                y.append(label)

        save_dataset(self.name, [np.array(X), np.array(y)])
        del X, y
        self.load()

    @staticmethod
    def audio(path):
        global SAMPLE_RATE
        return librosa.load(path, sr=SAMPLE_RATE, mono=True)

    @staticmethod
    def windows(
            signal,
            sample_rate,
            frame_size=0.025,
            frame_stride=0.01,
            pre_emphasis=0.97,
    ):
        """Author: Haytham Fayek"""
        emphasized_signal = np.append(signal[0],
                                      signal[1:] - pre_emphasis * signal[:-1])

        frame_length, frame_step = (
            frame_size * sample_rate,
            frame_stride * sample_rate,
        )  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(
            np.ceil(float(np.abs(signal_length - frame_length)) /
                    frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(
            emphasized_signal,
            z)  # Pad Signal to make sure that all frames have equal number of
        # samples without truncating any samples from the original signal

        indices = (np.tile(np.arange(0, frame_length), (num_frames, 1)) +
                   np.tile(
                       np.arange(0, num_frames * frame_step, frame_step),
                       (frame_length, 1),
                   ).T)
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames *= np.hamming(frame_length)

        return frames

    def features(signal):
        return fbanks(signal)


class Model:
    params = {
        "boosting_type": "dart",
        "objective": "binary",
        "metric": ["binary_error", "binary_logloss"],
        "is_unbalance": True,
        "feature_fraction": 0.85,
        # "learning_rate": 0.005,
        "learning_rate": 0.05,
        "verbose": -1,
        "min_split_gain": 0.1,
        "reg_alpha": 0.3,
        "max_bin": 7,
        "num_leaves": 7,
        "max_depth": 7,
        # "feature_fraction": 0.4,
        # "bagging_freq": 5,
        # "bagging_fraction": 0.4,
        # "max_bin": 10,
        # "num_leaves": 32,
        # "max_depth": 9,
        "min_child_weight": 0.5,
        # "num_iterations": 20,
        # "convert_model_language": "cpp",
    }

    def __init__(self, dataset=None):
        files = glob("data/trainall/*.wav")
        self.dataset = dataset
        # self.load()

    def samples(self):
        X_train, X_test, y_train, y_test = train_test_split(self.dataset.X,
                                                            self.dataset.y,
                                                            test_size=0.33,
                                                            random_state=42)

        print("X_train --->", X_train.shape)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_test = lgb.Dataset(X_test, y_test)
        return lgb_train, lgb_test

    def train(self):
        global MODEL_PATH

        lgb_train, lgb_test = self.samples()

        def learning_rate(epoch, span=100):
            cycle = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.005]
            lr = cycle[(epoch // span) % len(cycle)]
            print(f"LEARN RATE = {lr}")
            return lr

        gbm = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=4000,  # +oo
            valid_sets=lgb_test,
            # init_model=MODEL_PATH,
            early_stopping_rounds=500,  # 5000
            callbacks=[lgb.reset_parameter(learning_rate=learning_rate)],
        )
        gbm.save_model(MODEL_PATH)
        self.load()

    def load(self):
        self.pst = lgb.Booster(model_file=MODEL_PATH)

    def show(self):
        print("Feature importances:", list(self.pst.feature_importance()))

        for i in range(0, 1):
            ax = lgb.plot_tree(self.pst, tree_index=i)
            plt.show()

        ax = lgb.plot_importance(self.pst, importance_type="gain")
        plt.show()

    def predict(self, signal, sample_rate):
        windows = Dataset.windows(signal, sample_rate=sample_rate)
        X_predict = []
        skip = int(windows.shape[0] / (0.1 * windows.shape[0]))
        # print(windows.shape, skip)
        for window in windows[::skip]:
            features = Dataset.features(window)
            X_predict.append(features)
        pred = np.array(self.pst.predict(X_predict)).mean()
        if pred > 0.5:
            return "K"
        else:
            return "M"


def test(model):
    # weryfikacja dzialania
    global MODEL_PATH, SAMPLE_RATE, DATASET_FILES
    files = glob(DATASET_FILES)

    count_false = 0
    T1 = time.time()
    for path in tqdm(files):
        print(f"path={path}")
        signal, sample_rate = Dataset.audio(path)

        pred = model.predict(signal, sample_rate=sample_rate)
        if pred == path.split("_")[-1][0]:
            result = "\033[92m OKAY \033[m"
        else:
            result = "\033[90m;-( \033[m"
            count_false += 1
        print(f"path={path} | {pred} {result}")
    T2 = time.time()
    print(f"speed = {(T2-T1)/len(files)}sec")
    print(f"accuracy = {((len(files)-count_false)/len(files))*100}%")


def do(model, path):
    signal, sample_rate = Dataset.audio(path)
    pred = model.predict(signal, sample_rate=sample_rate)
    print(f"RESULT = {pred}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice.")
    parser.add_argument(
        "--dataset",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="dataset",
    )
    parser.add_argument("--train",
                        type=bool,
                        nargs="?",
                        const=True,
                        default=False,
                        help="train")
    parser.add_argument("--test",
                        type=bool,
                        nargs="?",
                        const=True,
                        default=False,
                        help="train")
    parser.add_argument("--do",
                        type=bool,
                        nargs="?",
                        const=True,
                        default=False,
                        help="do")

    parser.add_argument("--path",
                        type=str,
                        nargs="?",
                        const=True,
                        default=False,
                        help="do")
    args = parser.parse_args()
    print(args)
    files = glob(DATASET_FILES)

    if args.dataset:
        dataset = Dataset(files)
        dataset.create()
    if args.train:
        dataset = Dataset(files)
        model = Model(dataset=dataset)
        model.train()
        model.show()
    if args.test:
        model = Model()
        model.load()
        test(model)
    if args.do:
        model = Model()
        model.load()
        do(model, args.path)
