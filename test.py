import foa_scaler
import soundfile as sf
import numpy as np


#Fs = 48000
#T = 2.0
#f0 = 440.0

#t = np.arange(0, T, 1/Fs)
#sine= np.sin(2*np.pi*f0*t)

sine_sweep, fs = sf.read('Sine_Sweep.wav')
a_format = np.zeros((len(sine_sweep),4))
a_format[:, 0] = np.zeros(len(sine_sweep))
a_format[:, 1] = sine_sweep * 0.5
a_format[:, 2] = sine_sweep * 0.2
a_format[:, 3] = sine_sweep * 0.8

x_formatted = foa_scaler.convert_AtoB_tetramic(a_format)

x_scaled = foa_scaler.scale_Bformat_signals(x_formatted, 
                                            left_scaler = 2.0, 
                                            right_scaler = 1.0, 
                                            top_scaler = 1.0,
                                            bottom_scaler = 1.0,
                                            front_scaler = 1.0,
                                            back_scaler = 1.0,)
sf.write(f'/Volumes/Tom_2026/Left_Test_Sweep_No_W.wav', x_scaled, fs, format='WAV')