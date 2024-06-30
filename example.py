from importlib import reload
import librosa
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fft
import soundfile as sf

import chimera; reload(chimera)


# compare with http://research.meei.harvard.edu/chimera/speech_music.html

x1, sr = librosa.load('input1.wav')
x2, sr = librosa.load('input2.wav')

for bands in nbands:
    e1, e2 = chimera.generate(x1, x2, 128, sr) #High band count to take advantage of GPU acceleration

    sf.write('output1.wav', e1, sr)
    sf.write('output2.wav', e2, sr)
