# Usage:
# $ python3 bin/5.py --do --path='data/trainall/049_M.wav'
# $ sox -t alsa default recording.wav silence 1 0.1 5% 1 1.0 5%

################################################################################
# Used materials:
# https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
# https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
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
import pickle
import librosa
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
        # FIXME: okay, te wystarcza, przepisac na moja implt.
        from python_speech_features import fbank, ssc

        fbanks_l, fbanks_r = fbank(signal)
        features = [
            list(ssc(signal).mean(axis=0).flatten()),
            list(np.array(fbanks_l).mean(axis=0).flatten()),
            list(fbanks_r.flatten()),
        ]

        return np.array(sum(features, []))


class Model:
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": ["binary_error", "binary_logloss"],
        "is_unbalance": True,
        "feature_fraction": 0.85,
        "learning_rate": 0.005,
        "verbose": -1,
        "min_split_gain": 0.1,
        "reg_alpha": 0.3,
        "max_bin": 10,
        "num_leaves": 32,
        "max_depth": 9,
        "min_child_weight": 0.5,
    }

    def __init__(self, dataset=None):
        files = glob("data/trainall/*.wav")
        self.dataset = dataset
        self.load()

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

        gbm = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=100 * 50000,
            valid_sets=lgb_test,
            init_model=MODEL_PATH,
            early_stopping_rounds=2000,  # 5000
        )
        gbm.save_model(MODEL_PATH)
        self.load()

    def load(self):
        self.pst = lgb.Booster(model_file=MODEL_PATH)

    def show(self):
        print("Feature importances:", list(self.pst.feature_importance()))

        ax = lgb.plot_tree(self.pst)
        plt.show()

        ax = lgb.plot_importance(self.pst,
                                 importance_type="gain",
                                 max_num_features=30)
        plt.show()

    def predict(self, signal, sample_rate):
        windows = Dataset.windows(signal, sample_rate=sample_rate)
        X_predict = []
        skip = int(windows.shape[0] / (0.01 * windows.shape[0]))
        # print(windows.shape, skip)
        for window in windows[::skip]:
            features = Dataset.features(window)
            X_predict.append(features)
        pred = np.array(self.pst.predict(X_predict)).mean()
        if pred > 0.5:
            return "K"
        else:
            return "M"


# MANUAL TEST
def test(model):
    global MODEL_PATH, SAMPLE_RATE, DATASET_FILES
    files = glob(DATASET_FILES)

    for path in tqdm(files):
        print(f"path={path}")
        signal, sample_rate = Dataset.audio(path)

        pred = model.predict(signal, sample_rate=sample_rate)
        if pred == path.split("_")[-1][0]:
            result = "\033[92m OKAY \033[m"
        else:
            result = "\033[90m;-( \033[m"
        print(f"path={path} | {pred} {result}")


def do(model, path):
    signal, sample_rate = Dataset.audio(path)
    pred = model.predict(signal, sample_rate=sample_rate)
    print(f"RESULT = {pred}")


if __name__ == "__main__":
    import argparse

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
        test(model)
    if args.do:
        model = Model()
        do(model, args.path)
