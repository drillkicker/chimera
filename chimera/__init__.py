from __future__ import division

import cupy as cp
import scipy.signal as signal
from scipy.fftpack import fft, ifft

def generate(x1, x2, bands, sr, fmin=80, fmax=None):
    if fmax is None:
        fmax = 0.4 * sr

    fco = equal_xbm_bands(fmin, fmax, bands)
    e1, e2 = multi_band(x1, x2, fco, sr)
    return e1, e2

def inv_cochlear_map(x, fmax=None):
    f = (456 * (10**(0.021 * x)) - 364.8)
    if fmax is not None:
        f = f * fmax / inv_cochlear_map(100)
    return f

def cochlear_map(f, fmax=None):
    if fmax is not None:
        f = f * inv_cochlear_map(100) / fmax
    return cp.log10(f / 456 + 0.8) / 0.021

def equal_xbm_bands(fmin, fmax, n):
    xmin = cochlear_map(fmin, 20000)
    xmax = cochlear_map(fmax, 20000)
    x = cp.linspace(xmin, xmax, n + 1)
    return inv_cochlear_map(x, 20000)

def quad_filt_bank(cutoffs, fs=1):
    w = 2 * cp.divide(cutoffs, fs)
    n = int(cp.round(8 / cp.min(cp.diff(w))))
    n = n + 1 if n % 2 != 1 else n

    nbands = len(cutoffs) - 1
    b = cp.zeros((n + 1, nbands), dtype=complex)
    bw = cp.diff(w) / 2
    fo = w[0:nbands] + bw
    t = cp.arange(-n / 2, n / 2 + 1)

    for k in range(nbands):
        fir1 = signal.firwin(n + 1, cp.asnumpy(bw[k]))  # Convert to NumPy array
        cmplx = cp.exp(1j * cp.pi * fo[k] * t)
        b[:, k] = (cp.asarray(fir1) * cmplx).T  # Convert fir1 back to CuPy array

    return b

def multi_band(x1, x2, fco, fs=1):
    b = quad_filt_bank(fco, fs)

    # zero padding to match lengths
    if len(x1) < len(x2):
        x1 = cp.pad(x1, (0, len(x2) - len(x1)), 'constant', constant_values=(0,))
    elif len(x2) < len(x1):
        x2 = cp.pad(x2, (0, len(x1) - len(x2)), 'constant', constant_values=(0,))

    e1_fs2 = cp.zeros(x1.shape)
    e2_fs1 = cp.zeros(x1.shape)

    for k in range(b.shape[1]):
        zfilt1 = signal.lfilter(cp.asnumpy(b[:, k]), 1, cp.asnumpy(x1))  # Convert to NumPy arrays
        zfilt2 = signal.lfilter(cp.asnumpy(b[:, k]), 1, cp.asnumpy(x2))  # Convert to NumPy arrays

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
