# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
from helpers.io_methods import AudioIO as Io
from numpy.lib import stride_tricks
from helpers import tf_methods as tf
import numpy as np
import os

# definitions
dataset_path = '/Datasets/musdb18/'
test_dataset_path = '/Datasets/musdb18/'
keywords = ['bass.wav', 'drums.wav', 'other.wav', 'vocals.wav', 'mixture.wav']
foldersList = ['train', 'test']
save_path = 'results/'

__all__ = [
    'prepare_overlap_sequences',
    'get_data'
]


def prepare_overlap_sequences(ms, vs, bk, l_size, o_lap, bsize):
    """
        Method to prepare overlapping sequences of the given magnitude spectra.
        Args:
            ms               : (2D Array)  Mixture magnitude spectra (Time frames times Frequency sub-bands).
            vs               : (2D Array)  Singing voice magnitude spectra (Time frames times Frequency sub-bands).
            bk               : (2D Array)  Background magnitude spectra (Time frames times Frequency sub-bands).
            l_size           : (int)       Length of the time-sequence.
            o_lap            : (int)       Overlap between spectrogram time-sequences
                                           (to recover the missing information from the context information).
            bsize            : (int)       Batch size.

        Returns:
            ms               : (3D Array)  Mixture magnitude spectra training data
                                           reshaped into overlapping sequences.
            vs               : (3D Array)  Singing voice magnitude spectra training data
                                           reshaped into overlapping sequences.
            bk               : (3D Array)  Background magnitude spectra training data
                                           reshaped into overlapping sequences.

    """
    trim_frame = ms.shape[0] % (l_size - o_lap)
    trim_frame -= (l_size - o_lap)
    trim_frame = np.abs(trim_frame)
    # Zero-padding
    if trim_frame != 0:
        ms = np.pad(ms, ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))
        vs = np.pad(vs, ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))
        bk = np.pad(bk, ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))

    # Reshaping with overlap
    ms = stride_tricks.as_strided(ms, shape=(ms.shape[0] / (l_size - o_lap), l_size, ms.shape[1]),
                                  strides=(ms.strides[0] * (l_size - o_lap), ms.strides[0], ms.strides[1]))
    ms = ms[:-1, :, :]

    vs = stride_tricks.as_strided(vs, shape=(vs.shape[0] / (l_size - o_lap), l_size, vs.shape[1]),
                                  strides=(vs.strides[0] * (l_size - o_lap), vs.strides[0], vs.strides[1]))
    vs = vs[:-1, :, :]

    bk = stride_tricks.as_strided(bk, shape=(bk.shape[0] / (l_size - o_lap), l_size, bk.shape[1]),
                                  strides=(bk.strides[0] * (l_size - o_lap), bk.strides[0], bk.strides[1]))
    bk = bk[:-1, :, :]

    b_trim_frame = (ms.shape[0] % bsize)
    if b_trim_frame != 0:
        ms = ms[:-b_trim_frame, :, :]
        vs = vs[:-b_trim_frame, :, :]
        bk = bk[:-b_trim_frame, :, :]

    return ms, vs, bk


def get_data(current_set, set_size, wsz=2049, N=4096, hop=384, T=100, L=20, B=16):
    """
        Method to acquire training data. The STFT analysis is included.
        Args:
            current_set      : (int)       An integer denoting the current training set.
            set_size         : (int)       The amount of files a set has.
            wsz              : (int)       Window size in samples.
            N                : (int)       The FFT size.
            hop              : (int)       Hop size in samples.
            T                : (int)       Length of the time-sequence.
            L                : (int)       Number of context frames from the time-sequence.
            B                : (int)       Batch size.

        Returns:
            ms_train        :  (3D Array)  Mixture magnitude training data, for the current set.
            vs_train        :  (3D Array)  Singing voice magnitude training data, for the current set.

    """

    # Generate full paths for dev and test
    dev_list = sorted(os.listdir(dataset_path + foldersList[0]))
    dev_list = [dataset_path + foldersList[0] + '/' + i for i in dev_list]

    # Current lists for training
    c_train_mlist = dev_list[(current_set - 1) * set_size: current_set * set_size]

    for index in range(len(c_train_mlist)):
        # Reading
        print(c_train_mlist[index])
        vox, _ = Io.wavRead(os.path.join(c_train_mlist[index], keywords[3]), mono=True)
        mix, _ = Io.wavRead(os.path.join(c_train_mlist[index], keywords[4]), mono=True)

        # STFT Analysing
        ms_seg, _ = tf.TimeFrequencyDecomposition.STFT(mix, tf.hamming(wsz, True), N, hop)
        vs_seg, _ = tf.TimeFrequencyDecomposition.STFT(vox, tf.hamming(wsz, True), N, hop)

        # Remove null frames
        ms_seg = ms_seg[3:-3, :]
        vs_seg = vs_seg[3:-3, :]

        # Stack some spectrograms and fit
        if index == 0:
            ms_train = ms_seg
            vs_train = vs_seg
        else:
            ms_train = np.vstack((ms_train, ms_seg))
            vs_train = np.vstack((vs_train, vs_seg))

    # Freeing up some memory
    ms_seg = None
    vs_seg = None

    # Sequence creation
    ms_train, vs_train, _ = prepare_overlap_sequences(ms_train, vs_train, ms_train, T, L*2, B)

    return ms_train, vs_train


# EOF
