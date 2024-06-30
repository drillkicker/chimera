from __future__ import division

import cupy as cp
import scipy.signal as signal
from scipy.fftpack import fft, ifft


def generate(x1, x2, bands, sr, fmin=80, fmax=None):
    """Synthesize chimeras.

    :param x1: Original signal 1.
    :param x2: Original signal 2.
    :param bands: Number of frequency bands.
    :param sr: Sampling rate.
    :param fmin: (optional) Minimum frequency.
    :param fmax: (optional) Maximum frequency.
    :return: e1, e2 chimeras with envelope and fine structure of x1 or x2
        respectively.

    Usage::

        >>> import chimera
        >>> # load files e.g. with librosa
        >>> x1, sr = librosa.load('data/wood700us.wav')
        >>> x2, sr = librosa.load('data/take5-700us.wav')
        >>> e1, e2 = chimera.generate(x1, x2, 16, sr)
    """
    if fmax is None:
        fmax = 0.4 * sr

    fco = equal_xbm_bands(fmin, fmax, bands)
    e1, e2 = multi_band(x1, x2, fco, sr)
    return e1, e2


def inv_cochlear_map(x, fmax=None):
    """Convert distance along the basilar membrane to frequency using M.C.
    Liberman's cochlear frequency map for the cat.

    :param x: Percent distance from apex of basilar membrane.
    :param fmax: Maximum frequency represented on the basilar membrane in Hz.
        By default, this is 57 kHz, the value for the cat.
        Setting Fmax to 20 kHz gives a map appropriate for the human cochlea.

    :return: Frequency in Hz.
    """
    f = (456 * (10**(0.021 * x)) - 364.8)
    if fmax is not None:
        f = f * fmax / inv_cochlear_map(100)

    return f


def cochlear_map(f, fmax=None):
    """Convert frequency to distance along the basilar membrane. using M.C.
    Liberman's cochlear frequency map for the cat.

    :param f: Frequency in Hz.
    :param fmax: Maximum frequency represented on the basilar membrane in Hz.
        By default, this is 57 kHz, the value for the cat.
        Setting Fmax to 20 kHz gives a map appropriate for the human cochlea.

    :return: Percent distance from apex of basilar membrane.
    """
    if fmax is not None:
        f = f * inv_cochlear_map(100) / fmax

    return cp.log10(f / 456 + 0.8) / 0.021


def equal_xbm_bands(fmin, fmax, n):
    """Divide frequency interval into N bands of equal width along the human
    basilar membrane.

    :param fmin: Minimum frequency in Hz.
    :param fmax: Maximum frequency in Hz.
    :param n: Number of frequency bands.
    :return: Vector of band cutoff frequencies in Hz.
    """
    xmin = cochlear_map(fmin, 20000)
    xmax = cochlear_map(fmax, 20000)
    x = cp.linspace(xmin, xmax, n + 1)
    return inv_cochlear_map(x, 20000)


def quad_filt_bank(cutoffs, fs=1):
    """Create a bank of FIR complex filters whose real and imaginary parts are
    in quadrature.

    :param cutoffs: Band cutoff frequencies.
    :param fs: Sampling rate.
    :return: Filter bank.
    """
    # normalize frequency
    w = 2 * cp.divide(cutoffs, fs)
    n = int(cp.round(8 / cp.min(cp.diff(w))))
    n = n + 1 if n % 2 != 1 else n

    nbands = len(cutoffs) - 1
    b = cp.zeros((n + 1, nbands), dtype=complex)
    bw = cp.diff(w) / 2  # lowpass filter bandwidths
    fo = w[0:nbands] + bw  # freq offset
    t = cp.arange(-n / 2, n / 2 + 1)  # time vector

    for k in range(nbands):
        fir1 = signal.firwin(n + 1, bw[k].get())  # Convert to NumPy array
        cmplx = cp.exp(1j * cp.pi * fo[k] * t)
        b[:, k] = (cp.asarray(fir1) * cmplx).T  # Convert fir1 back to CuPy array

    return b


def multi_band(x1, x2, fco, fs=1):
    """Synthesize pair of multi-band "auditory chimeras" by dividing each
    signal into frequency bands, then interchanging envelope and fine
    structure in each band using Hilbert transforms.

    :param x1: Original signal 1.
    :param x2: Original signal 2.
    :param fco: Band cutoff frequencies or filter bank.
    :param fs: Sampling rate in Hz.

    :return: e1, e2 chimeras with envelope and fine structure of x1 or x2
        respectively.
    """
    b = quad_filt_bank(fco, fs)

    # Ensure the signals are cupy arrays
    x1 = cp.asarray(x1)
    x2 = cp.asarray(x2)

    # zero padding to match lengths
    if len(x1) < len(x2):
        x1 = cp.pad(x1, (0, len(x2) - len(x1)), 'constant', constant_values=(0,))
    elif len(x2) < len(x1):
        x2 = cp.pad(x2, (0, len(x1) - len(x2)), 'constant', constant_values=(0,))

    e1_fs2 = cp.zeros(x1.shape)
    e2_fs1 = cp.zeros(x1.shape)

    for k in range(b.shape[1]):
        # this is slow - matlab version uses fftfilt here, which does not
        # exist in scipy
        zfilt1 = signal.lfilter(b[:, k].get(), 1, x1.get())  # Convert to NumPy arrays
        zfilt2 = signal.lfilter(b[:, k].get(), 1, x2.get())  # Convert to NumPy arrays

        # Convert back to CuPy arrays
        zfilt1 = cp.asarray(zfilt1)
        zfilt2 = cp.asarray(zfilt2)

        # switch envelope and fine structure
        e1_fs2_filt = cp.abs(zfilt1) * cp.cos(cp.angle(zfilt2))
        e2_fs1_filt = cp.abs(zfilt2) * cp.cos(cp.angle(zfilt1))

        # accumulate over frequency bands
        e1_fs2 += e1_fs2_filt
        e2_fs1 += e2_fs1_filt

    return e1_fs2, e2_fs1


def matched_noise(x):
    return cp.real(
        ifft(cp.abs(fft(x)) * cp.exp(1j * 2 * cp.pi * cp.random.rand(len(x)))))
