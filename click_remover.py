import numpy as np
from scipy.io import wavfile
from singer import play
import pdb

#this actually doesn't work very well...

FS, clip = wavfile.read('output/When David Heard.wav')

q1, q3 = np.percentile(clip, (25, 75))

iqr = q3 - q1

scale = 1.5

non_outliers = np.logical_and(clip < q3 + scale * iqr, clip > q1 - scale * iqr)
threshold = np.abs(clip[non_outliers]).max()

clean = np.tanh(clip / threshold) * threshold

wavfile.write(f'clean{scale}.wav', FS, clean)