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

    def __init__(self, singer_name='matt', FS_out=192000, mode='lpc'):#FS_out=192000, mode='sample'):
        self.name = singer_name
        
        self.FS_in = None
        self.FS_out = FS_out

        assert(mode in ['sample', 'lpc'])
        self.mode = mode
        
        self.phonemes = []  #list of phonemes that this voice can use. should set up a default phoneme for when one is requested that doesn't exist
        self.default_phoneme = 'u'
        with open('phonemes.json') as f:
            self.phoneme_letters = json.load(f) #list of the utf-8 characters that are recognized phonemes
        
        self.templates = {}
        self.lcrosspoints = {}
        self.ucrosspoints = {}
        self.duration_error = 0 #amount of error in duration sung vs requested be sung
        
        self.lpc = {} #coefficients and gains for lpc synthesis
        self.lpc_order = None

        if self.mode == 'sample': 
            self.load_samples()
        else:
            self.load_lpc()


    def load_samples(self):
        """load single period waveforms for the specified singer"""
        sample_folder = os.path.join('phoneme_data', 'single_period', self.name)
        for subddir, dirs, files in os.walk(sample_folder):
            for filename in files:
                if os.path.splitext(filename)[-1] == '.wav':
                    name, ext = os.path.splitext(filename)
                    rate, sample = wavfile.read(os.path.join(sample_folder, filename))
                    sample = np.concatenate((np.array([sample[-1]], dtype=np.float32), sample, np.array([sample[0]], dtype=np.float32))) #add the repeat sample to the end of the array
                    spline = CubicSpline(np.arange(sample.shape[0]), sample)
                    roots = spline.roots()
                    length = sample.shape[0]
                    lower = roots[0] if roots[0] > 0 else roots[1]
                    upper = roots[-2] if roots[-2] - (length - 2) > 0 else roots[-1]


                    if self.FS_in is None:     #set default sample rate, or verify that it's the same
                        self.FS_in = rate
                    else:
                        assert(rate == self.FS_in)

                    #save the sample plus its crossover point to the object
                    self.phonemes.append(name)
                    self.templates[name] = sample
                    self.lcrosspoints[name] = lower
                    self.ucrosspoints[name] = upper

        #make template entries for aliased phonemes (e.g. j is pronounced with i, and approximate w is pronounced with u)
        aliases = self.phoneme_letters['aliases']
        for p, a in aliases.items():
            self.templates[p] = self.templates[a]
            self.lcrosspoints[p] = self.lcrosspoints[a]
            self.ucrosspoints[p] = self.ucrosspoints[a]


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
        for phoneme, alias in self.phoneme_letters['aliases'].items():
            self.lpc[phoneme] = self.lpc[alias]
            self.phonemes.append(phoneme)

        #silent/unpitched phoneme lpc parameters
        self.lpc['0'] = {
            'order': self.lpc_order,
            'coeffs': np.zeros(self.lpc_order, dtype=np.float64),
            'gain': np.zeros(1, dtype=np.float64),
        }



    def sing_excerpt(self, excerpt):
        if self.mode == 'sample':
            return self.sing_excerpt_sample_mode(excerpt)
        elif self.mode == 'lpc':
            return self.sing_excerpt_lpc_mode(excerpt)
        else:
            raise Exception(f'ERROR: {self.mode} mode is not a valid mode for the singer')


    def sing_excerpt_lpc_mode(self, excerpt):
        # num_samples = sum([int(note['duration'] * self.FS_out) for note in excerpt])
        num_samples = sum(int(note.duration * self.FS_out) for note in excerpt)

        sample_t = np.arange(num_samples, dtype=np.float64) / self.FS_out #time each sample occurs at
        # volumes = np.concatenate([np.ones(int(note.duration * self.FS_out), dtype=np.float64) * note.volume for note in excerpt]) #TODO
        volumes = np.ones_like(sample_t) * 0.5 #for now all volume is 50%    
        pitch = np.concatenate([np.ones(int(note.duration * self.FS_out), dtype=np.float64) * (note.pitch if note.is_sung() else 0.00000000001) for note in excerpt])
        
        phonemes, gains = zip(*[self.get_phoneme_gain_sequences(note) for note in excerpt])
        phonemes = np.concatenate(phonemes)
        gains = np.concatenate(gains)

        # phonemes = np.concatenate([self.get_phoneme_sequence(note) for note in excerpt])
        # gains = np.concatenate([self.get_phoneme_gains(note) for note in excerpt])

        #get buzz and noise. combine into a single source
        buzz, noise = pitched_squareDC(pitch, 0.25, sample_t), white_noise_t(sample_t)
        source = 1.5 * buzz + 0.3 * noise

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



    # def get_phoneme_sequence(self, note):
    #     """create an array of unicode characters representing the phoneme at each instant in the note"""
    #     num_samples = int(note['duration'] * self.FS_out)
    #     sequence = np.chararray(num_samples, unicode=True)

    #     sequence[:] = '0' #set all unset phonemes to silent
        
    #     if note['volume'] != 0:
            
    #         count = 0
    #         phoneme_duration_pairs = self.partition_syllable(note['syllable'], note['duration'])
    #         for phoneme, duration in phoneme_duration_pairs:
    #             if phoneme not in self.phonemes:
    #                 phoneme = self.default_phoneme
    #             phoneme_samples = int(note['duration'] * self.FS_out)
    #             sequence[count:count+phoneme_samples] = phoneme
    #             count += phoneme_samples
    #         sequence[count:] = phoneme_duration_pairs[-1][0] #set any remaining samples to the last phoneme in the sequence

    #     return sequence

    # def get_phoneme_gains(self, note):
    #     """create an array of gains for each phoneme at each instant in the note"""
    #     num_samples = int(note['duration'] * self.FS_out)
    #     gains = np.zeros(num_samples, dtype=np.float64)

    #     if note['volume'] != 0:
    #         assert(len(note['syllable']) == 1) #TODO->allow multiple phoneme syllable.
    #         phoneme = note['syllable'] if note['syllable'] in self.phonemes else self.default_phoneme

    #         #because not splitting up note into multiple phonemes at the moment, just assign the phoneme to the whole sample
    #         gains[:] = self.lpc[phoneme]['gain']

    #     return gains


    def get_phoneme_gain_sequences(self, note):
        """create an array of unicode characters representing the phoneme at each instant in the note"""
        num_samples = int(note.duration * self.FS_out)

        sequence = np.chararray(num_samples, unicode=True)
        sequence[:] = '0' #set all unset phonemes to silent

        gains = np.zeros(num_samples, dtype=np.float64)
        
        if note.is_sung():            
            count = 0 #current sample position in the arrays
            for phoneme, duration in self.clock_phonemes(note.phonemes, note.duration):
                if phoneme not in self.phonemes:
                    print(f'unrecognized phoneme [{phoneme}]')
                    phoneme = self.default_phoneme
                phoneme_samples = int(duration * self.FS_out)
                sequence[count:count+phoneme_samples] = phoneme
                gains[count:count+phoneme_samples] = self.lpc[phoneme]['gain']

                count += phoneme_samples

            sequence[count:] = phoneme
            gains[count:] = self.lpc[phoneme]['gain']
        return sequence, gains

    # def get_phoneme_lpc_arrays(self, note):
    #     """create an array for the phoneme's lpc gains and coefficients"""
    #     samples = int(note['duration'] * self.FS_in)
    #     coeffs = np.zeros((samples, self.lpc_order), dtype=np.float64)
    #     gains = np.zeros(samples, dtype=np.float64)

    #     if note['volume'] != 0:
    #         assert(len(note['syllable']) == 1) #TODO->allow multiple phoneme syllable.

    #         phoneme = 'a' if note['syllable'] not in self.phonemes else note['syllable']
    #         gains[:] = self.lpc[phoneme]['gain']
    #         coeffs[None, :] = self.lpc[phoneme]['coeffs']

    #     return gains, coeffs
        


    def sing_excerpt_sample_mode(self, excerpt):
        note_samples = []
        for note in excerpt:
            note_samples.append(self.sing_note(note))

        start = 0
        sample_length = sum([note_sample.shape[0] for note_sample in note_samples])
        sample = np.zeros(sample_length)
        for note_sample in note_samples:
            sample[start:start+note_sample.shape[0]] = note_sample
            start += len(note_sample)

        # self.duration_error = 0
        return sample


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


    # def partition_syllable(self, syllable, duration):
    #     #split the syllable into durations for each phoneme
    #     #TODO: use rates from http://www.asel.udel.edu/icslp/cdrom/vol4/301/a301.pdf

    #     consonants = self.phoneme_letters['consonants']
    #     vowels = self.phoneme_letters['vowels']

    #     #verify that the word is made of correct characters
    #     for p in syllable:
    #         assert p in consonants or p in vowels

    #     #break the syllable into an [optional] initial consonant (attack), middle vowel (sustain), and [optional] ending consonant (release)
    #     attack, sustain, release = '', '', ''

    #     for p in syllable:
    #         if p not in consonants:
    #             break
    #         attack += p

    #     for p in syllable[len(attack):]:
    #         if p not in vowels:
    #             break
    #         sustain += p

    #     for p in syllable[len(attack+sustain):]:
    #         if p not in consonants:
    #             break
    #         release += p

    #     assert attack + sustain + release == syllable   #we should have successfuly split the entire syllable
    #     assert len(sustain) > 0     #syllabars are of the form {C},V,{C}, i.e. sustain must contian a vowel

    #     #compute the duration of each portion of the syllable
    #     attack_duration = 0 if len(attack) == 0 else min(0.15, 0.25*duration)
    #     release_duration = 0 if len(release) == 0 else min(0.15, 0.25*duration)
    #     sustain_duration = duration - attack_duration - release_duration

    #     #convert the three syllable portions into a single list of phonemes and durations
    #     phoneme_duration_pairs = []
    #     for group, duration in zip((attack, sustain, release), (attack_duration, sustain_duration, release_duration)):
    #         for p in group:
    #             phoneme_duration_pairs += [(p, duration / len(group))] #evenly split time between each component of the syllable group

    #     return phoneme_duration_pairs


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
