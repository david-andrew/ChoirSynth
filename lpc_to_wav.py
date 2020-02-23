import sys
import pdb
# from random import random as rand
import simpleaudio as sa
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt
import time


def main():
    order, sample_period, num_frames, frame_length, frame_coeffs, frame_gains = parse_LPC(sys.argv[1])
    # t0 = time.time()
    # samples_slow = generate_wav_slow(order, sample_period, num_frames, frame_length, frame_coeffs, frame_gains)
    # print(f'slow time: {time.time() - t0}')
    t0 = time.time()
    samples_fast = generate_wav(order, sample_period, num_frames, frame_length, frame_coeffs, frame_gains)
    print(f'fast time: {time.time() - t0}')
    # play(samples_slow)
    play(samples_fast)




def play(sample, FS=44100, block=True):
    if sample.dtype in [np.float32, np.float64]: #convert to 16-bit integer
        sample = (sample * np.iinfo(np.int16).max).astype(np.int16)
    assert sample.dtype == np.int16
    player = sa.play_buffer(audio_data=sample, num_channels=1, bytes_per_sample=2, sample_rate=FS)
    if block:
        player.wait_done()

# vector functions for producing various waveforms at the specified pitch
# pitch p (Hz) at time t (s). pitched_squareDC takes a parameter DC for duty cycle
pitched_sawtooth = np.vectorize(lambda p, t: (t % (1 / p)) * p - 0.5)
pitched_square =   np.vectorize(lambda p, t: -1.0 if ((t % (1 / p)) * p) < 0.5 else 1.0)
pitched_squareDC = np.vectorize(lambda p, DC, t: -1.0 if ((t % (1 / p)) * p) < DC else 1.0)
pitched_triangle = np.vectorize(lambda p, t: 4 * ((t % (1 / p)) * p) - 1.0 if ((t % (1 / p)) * p) < 0.5 else -4 * (((t % (1 / p)) * p) - 0.5) + 1.0)
pitched_sin = lambda p, t: np.sin(((t % (1 / p)) * p) * 2 * np.pi)
white_noise_t = np.vectorize(lambda t: np.random.random() * 2 - 1.0)
# white_noise_n = lambda n: np.random.random(n) * 2 - 1.0




def parse_LPC(path):
    with open(path, 'r') as f:
        data = f.read()

    lines = data.split('\n')

    #parse the different lines of the file
    xmin = float(lines[3].split('= ')[1])
    xmax = float(lines[4].split('= ')[1])
    num_frames = int(lines[5].split('= ')[1])
    frame_length = float(lines[6].split('= ')[1])
    # x1 = float(lines[7].split('= ')[1])
    sample_period = float(lines[8].split('= ')[1])
    order = int(lines[9].split('= ')[1])

    frame_lines = lines[11:] if lines[-1] != '' else lines[11:-1] #remove last line if empty

    l = len(frame_lines)
    n = int(l / num_frames); 
    assert(n == l/num_frames) #verify that the file is being split up correctly. every frame should have the same number of lines, thus this should be a whole number
    frames_chunks = [frame_lines[i:i+n] for i in range(0, l, n)]

    frame_gains = np.array([float(frame[-1].split('= ')[1]) for frame in frames_chunks])
    frame_coeffs = np.array([[float(cline.split('= ')[1]) for cline in frame[3:-1]] for frame in frames_chunks])
    assert(order == len(frame_coeffs[0])) #verify that the order we recorded from the file is the same as the number of coefficients we extracted from each frame

    return order, sample_period, num_frames, frame_length, frame_coeffs, frame_gains


def generate_wav_slow(order, sample_period, num_frames, frame_length, frame_coeffs, frame_gains):
    rate = int(1 / sample_period)
    samples_per_frame = int(frame_length / sample_period)

    voice_pitch = 105
    bp = [0] * order
    samples = []
    count = 0
    offset = 0
    #construct the audio from the frame coefficients
    for coeffs, gain in zip(frame_coeffs, frame_gains):
        for sample in range(samples_per_frame):
            count %= int(rate / voice_pitch)
            w = count / (rate / voice_pitch)
            count += 1
            f = (np.random.random() - 0.5) * 0.5 + 2**w - 1 / (1 + w)

            acc = f
            for j in range(order):
                acc -= coeffs[j] * bp[(offset + order - j) % order]

            offset = (offset + 1) % order
            bp[offset] = acc
            samples.append(acc * gain**0.5)

    return np.array(samples)

def generate_wav(order, sample_period, num_frames, frame_length, frame_coeffs, frame_gains):
    #generate frame time array
    #generate output time array
    #interpolate gains, and coefficients for each output time

    FS = int(1 / sample_period)
    frame_rate = int(1 / frame_length)

    assert(FS == 1 / sample_period)
    assert(frame_rate == 1 / frame_length)

    frame_t = np.arange(num_frames) * frame_length
    
    num_samples = int(num_frames * frame_length * FS)
    sample_t = np.arange(num_samples) * sample_period

    interp_kwargs = lambda arr: {
        'x': frame_t,
        'y': arr,
        'fill_value': (arr[0], arr[-1]),
        'copy': False, 
        'assume_sorted': True, 
        'bounds_error': False,
        'kind': 'nearest' #other types of interpolation
    }

    interp_gains = interpolate.interp1d(**interp_kwargs(frame_gains))
    interp_coeffs = interpolate.interp1d(**interp_kwargs(frame_coeffs), axis=0)

    sample_gains = interp_gains(sample_t)
    sample_coeffs = interp_coeffs(sample_t)

    voice_pitch = 105
    buzz, noise = pitched_sawtooth(voice_pitch, sample_t), white_noise_t(sample_t)
    carrier = 1.5 * buzz + 0.5 * noise  #np.power(2, buzz) - 1/(1+buzz) + 0.5 * noise
    samples = np.zeros(num_samples + order, dtype=np.float64)

    #TODO->figure out how to do this without a loop?    
    rev_buffer = np.arange(0, -order, -1) #used to index into the output buffer in reverse order
    for j, (sample, coeffs) in enumerate(zip(carrier, sample_coeffs)):
        samples[order + j] = sample - (samples[rev_buffer + order + j - 1 ] @ coeffs)
    
    #apply gain to generated samples
    samples = samples[order:] * np.sqrt(sample_gains)
    return samples



if __name__ == '__main__':
    main()


