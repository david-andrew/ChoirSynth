import pdb
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import CubicSpline
import simpleaudio as sa
import os
from math import ceil, floor
from fractions import Fraction as frac
import time
import json
from lpc_to_wav import parse_lpc, pitched_sawtooth, pitched_square, pitched_squareDC, pitched_triangle, white_noise_t
from staticnote import StaticSyllable


#todo implement base class for sound generating objects.

#TODO->adjust template concatenation so that freequencies don't have to be a whole number multiple of the sampling frequency
#----->instead the template will phase shift, allowing for in-between frequencies to occur

class singer():

    def __init__(self, singer_name='matt', FS_out=192000):#FS_out=192000, mode='sample'):
        self.name = singer_name
        
        self.FS_in = None
        self.FS_out = FS_out
        
        self.phonemes = []  #list of phonemes that this voice can use. should set up a default phoneme for when one is requested that doesn't exist
        self.default_phoneme = 'u'
        with open('phonetic_dictionary.json') as f:
            phonetic_dictionary = json.load(f)['english']['u.s.'] #list of the utf-8 characters that are recognized phonemes
            self.phoneme_aliases = phonetic_dictionary['aliases']
            self.unvoiced_phonemes = set(phonetic_dictionary['unvoiced'])
            # self.vowel_phonemes = set(phonetic_dictionary['vowels'])
            # self.consonant_phonemes = set(phonetic_dictionary['consonants'])
            # self.dipthong_phonemes = set(phonetic_dictionary['dipthongs'])


        self.templates = {}
        self.lcrosspoints = {}
        self.ucrosspoints = {}
        self.duration_error = 0 #amount of error in duration sung vs requested be sung
        
        self.lpc = {} #coefficients and gains for lpc synthesis
        self.lpc_order = None


        self.load_lpc()


    def load_lpc(self):
        lpc_folder = os.path.join('phoneme_data', 'LPC', self.name)
        for subddir, dirs, files in os.walk(lpc_folder):
            for filename in files:
                if os.path.splitext(filename)[-1] == '.LPC':
                    phoneme, ext = os.path.splitext(filename)
                    # order, sample_period, num_frames, frame_length, frame_coeffs, frame_gains = parse_lpc(os.path.join(lpc_folder, filename))
                    order, _, _, _, frame_coeffs, frame_gains = parse_lpc(os.path.join(lpc_folder, filename))
                    self.phonemes.append(phoneme)
                    self.lpc[phoneme] = {
                        'order': order,
                        'coeffs': frame_coeffs.mean(axis=0),
                        'gain': frame_gains.mean(axis=0),
                    }
                    if self.lpc_order is None:
                        self.lpc_order = order
                    else:
                        assert(order == self.lpc_order)

        #set up alias phonemes
        for phoneme, alias in self.phoneme_aliases.items():
            self.lpc[phoneme] = self.lpc[alias]
            self.phonemes.append(phoneme)

        #silent/unpitched phoneme lpc parameters
        self.lpc['0'] = {
            'order': self.lpc_order,
            'coeffs': np.zeros(self.lpc_order, dtype=np.float64),
            'gain': np.zeros(1, dtype=np.float64),
        }
        

    def sing_excerpt(self, excerpt):
        # num_samples = sum([int(note['duration'] * self.FS_out) for note in excerpt])
        num_samples = sum(int(note.duration * self.FS_out) for note in excerpt)

        sample_t = np.arange(num_samples, dtype=np.float64) / self.FS_out #time each sample occurs at
        # volumes = np.concatenate([np.ones(int(note.duration * self.FS_out), dtype=np.float64) * note.volume for note in excerpt]) #TODO
        volumes = np.ones_like(sample_t) * 0.5 #for now all volume is 50%    
        pitch = np.concatenate([np.ones(int(note.duration * self.FS_out), dtype=np.float64) * (note.pitch if note.is_sung() else 0.00000000001) for note in excerpt])

        phonemes, gains, phonations = zip(*[self.sequence_phonetics(note) for note in excerpt])
        phonemes, gains, phonations = np.concatenate(phonemes), np.concatenate(gains), np.concatenate(phonations)

        # phonemes = np.concatenate([self.get_phoneme_sequence(note) for note in excerpt])
        # gains = np.concatenate([self.get_phoneme_gains(note) for note in excerpt])

        #get buzz and noise. combine into a single source
        buzz, noise = pitched_squareDC(pitch, 0.25, sample_t), white_noise_t(sample_t)
        source = 1.5 * buzz * phonations + 0.3 * noise #combine noise with buzz. mute buzz when unvoiced

        samples = np.zeros(num_samples + self.lpc_order, dtype=np.float64) #preallocate samples array. pad with 'order' 0s before the start of the sample output, so that the filter draws from them before we have generated 'order' samples 

        #TODO->figure out how to do this without a loop?    
        rev_buffer = np.arange(0, -self.lpc_order, -1) #used to index into the output buffer in reverse order
        for j, (sample, phoneme) in enumerate(zip(source, phonemes)):
            coeffs = self.lpc[phoneme]['coeffs']
            samples[self.lpc_order + j] = sample - (samples[rev_buffer + self.lpc_order + j - 1 ] @ coeffs)
        
        #apply gain to generated samples, and remove the zero padding
        samples = samples[self.lpc_order:] * np.sqrt(gains)
        samples = samples * volumes
        # pdb.set_trace()
        return samples


    def sequence_phonetics(self, note):
        """create an array of unicode characters representing the phoneme at each instant in the note"""
        num_samples = int(note.duration * self.FS_out)

        sequence = np.chararray(num_samples, unicode=True)
        sequence[:] = '0' #set all unset phonemes to silent

        gains = np.zeros(num_samples, dtype=np.float64)
        phonations = np.zeros(num_samples, dtype=np.float64) #when is it voiced vs unvoiced
        
        if note.is_sung():            
            count = 0 #current sample position in the arrays
            for phoneme, duration in self.clock_phonemes(note.phonemes, note.duration):
                if phoneme not in self.phonemes:
                    print(f'unrecognized phoneme [{phoneme}]')
                    phoneme = self.default_phoneme
                phoneme_samples = int(duration * self.FS_out)
                sequence[count:count+phoneme_samples] = phoneme
                gains[count:count+phoneme_samples] = self.lpc[phoneme]['gain']
                phonations[count:count+phoneme_samples] = 0.0 if phoneme in self.unvoiced_phonemes else 1.0

                count += phoneme_samples

            sequence[count:] = phoneme
            gains[count:] = self.lpc[phoneme]['gain']
            phonations[count:] = 0.0 if phoneme in self.unvoiced_phonemes else 1.0
        return sequence, gains, phonations

 
    def sing_note(self, note):
        #for now we are assuming all notes have just a single syllable. later on we will have to handle phoneme cluster cases i.e. dipthongs and tripthings. In the cases where there are more phonemes, all sounds are divided equally with their time
        duration = note['duration']
        volume = note['volume']
        
        if volume == 0: #indicates rest
            requested_duration = duration - self.duration_error
            true_duration = max(0, requested_duration) #rest duration must be at positive
            sample = np.zeros(int(self.FS_out * true_duration))
            self.duration_error = true_duration - requested_duration
            return sample
        
        else: #regular note to sing
            pitch = note['pitch']
            syllable = note['syllable']

            phoneme_samples = []

            for phoneme, duration in self.partition_syllable(syllable, duration):
                requested_duration = duration - self.duration_error
                repeat = max(1, ceil(requested_duration * pitch)) #make at least 1 period of the sound
                true_duration = repeat / pitch
                self.duration_error = true_duration - requested_duration

                period_samples = ceil(self.FS_out / pitch) #number of samples in a single period of the desired pitch
                output = np.zeros(period_samples * repeat)
                template = self.templates[phoneme]

                #linearly sample the template spline to set the template to the desired pitch
                spline = CubicSpline(np.arange(template.shape[0]), template)
                pitched_template = spline(np.linspace(self.lcrosspoints[phoneme], self.ucrosspoints[phoneme], period_samples)) 
                phoneme_samples.append(np.tile(pitched_template, repeat))
                           
            return np.concatenate(phoneme_samples)


    def clock_phonemes(self, phonemes, duration):
        # compute the duration of each portion of the syllable
        if phonemes is None:
            phonemes = StaticSyllable(sustain=self.default_phoneme)
        attack_duration = 0 if len(phonemes.attack) == 0 else min(frac(3, 20), frac(1, 4) * duration)
        release_duration = 0 if len(phonemes.release) == 0 else min(frac(3, 20), frac(1, 4) * duration)
        sustain_duration = duration - attack_duration - release_duration
        
        #convert the three syllable portions into a single list of phonemes and durations
        phoneme_duration_pairs = []
        for group, duration in zip((phonemes.attack, phonemes.sustain, phonemes.release), (attack_duration, sustain_duration, release_duration)):
            for p in group:
                phoneme_duration_pairs += [(p, duration / len(group))] #evenly split time between each component of the syllable group

        return phoneme_duration_pairs


def play(sample, FS=44100, block=True):
    if sample.dtype in [np.float32, np.float64]: #convert to 16-bit integer
        sample = (sample * np.iinfo(np.int16).max).astype(np.int16)
    assert sample.dtype == np.int16
    player = sa.play_buffer(audio_data=sample, num_channels=1, bytes_per_sample=2, sample_rate=FS)
    if block:
        player.wait_done()



if __name__ == '__main__':
    #demo of singing arpeggios of alleluia
    FS_out = 44100#192000
    
    matt = singer(singer_name='matt', FS_out=FS_out)
    
    pitches = [16.35, 17.32, 18.35, 19.45, 20.6, 21.83, 23.12, 24.5, 25.96, 27.5, 29.14, 30.87, 32.7, 34.65, 36.71, 38.89, 41.2, 43.65, 46.25, 49, 51.91, 55, 58.27, 61.74, 65.41, 69.3, 73.42, 77.78, 82.41, 87.31, 92.5, 98, 103.83, 110, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185, 196, 207.65, 220, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392, 415.3, 440, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880, 932.33, 987.77, 1046.5, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760, 1864.66, 1975.53, 2093, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520, 3729.31, 3951.07, 4186.01, 4434.92, 4698.63, 4978.03, 5274.04, 5587.65, 5919.91, 6271.93, 6644.88, 7040, 7458.62, 7902.13]
    
    for pitch in pitches:
        speed = 0.2
        pitch = np.random.uniform(20, 260)
        print(pitch)
        notes = [
            {'syllable':'a', 'duration':1*speed, 'pitch':pitch*np.power(2,12/12), 'volume':1},
            {'syllable':'le', 'duration':1*speed, 'pitch':pitch*np.power(2,7/12), 'volume':1},
            {'syllable':'lu', 'duration':1*speed, 'pitch':pitch*np.power(2,4/12), 'volume':1},
            {'syllable':'ja', 'duration':2*speed, 'pitch':pitch*np.power(2,0/12), 'volume':1}
        ]
        play(matt.sing_excerpt(notes), FS_out)

        time.sleep(.2)
