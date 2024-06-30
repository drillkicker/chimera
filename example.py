from importlib import reload
import librosa
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fft
import soundfile as sf
import chimera
import cupy as cp

reload(chimera)

# compare with http://research.meei.harvard.edu/chimera/speech_music.html

x1, sr = librosa.load('input1.wav')
x2, sr = librosa.load('input2.wav')

# Convert audio signals to CuPy arrays
x1 = cp.asarray(x1)
x2 = cp.asarray(x2

e1, e2 = chimera.generate(x1, x2, 128, sr) #GPU acceleration allows us to use a high band count without taking an immense amount of time

# Convert CuPy arrays to NumPy arrays before saving
e1_np = cp.asnumpy(e1)
e2_np = cp.asnumpy(e2)

sf.write('ooutput1.wav', e1_np, sr)
sf.write('output2.wav', e2_np, sr)
