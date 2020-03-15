import pdb

import os
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
import music21
from math import gcd
import numpy as np
from scipy.io import wavfile
from fractions import Fraction as frac
# from numpy import lcm
import json
from staticnote import StaticNote, StaticSyllable


#parse musicxml sheet music and convert to individual excerpts for each singer
#TODO:
# for now we're skipping chords, as well as multiple voices on a line

language = 'english'
accent = 'u.s.'

with open('phonetic_dictionary.json') as f:
    raw_phonetic_dictionary = json.load(f)
    phonetic_dictionary = raw_phonetic_dictionary[language][accent]['words']
    vowels = raw_phonetic_dictionary[language][accent]['vowels']
    diphthongs = raw_phonetic_dictionary[language][accent]['diphthongs']
    consonants = raw_phonetic_dictionary[language][accent]['consonants']
    aliases = raw_phonetic_dictionary[language][accent]['aliases']


default_phoneme = 'u'


class PartStream():
    def __init__(self, part_stream, max_voice_splits):
        self.part_stream = part_stream
        self.max_voice_splits = max_voice_splits

    def __iter__(self):
        return PartStreamIterator(self.part_stream, self.max_voice_splits)

    def spread_iter(self):
        return PartStreamIterator(self.part_stream, self.max_voice_splits, spread=True)

    def __str__(self):
        out = ''
        for stream_stack in self:
            out += f'[{", ".join([str(note) for note in stream_stack])}]\n'

        return out


class PartStreamIterator():
    def __init__(self, part_stream, max_voice_splits, spread=False):
        self.i = 0
        self.offset = frac(0)
        self.part_stream = part_stream
        self.max_voice_splits = max_voice_splits
        self.current = [None] * max_voice_splits
        self.spread = spread

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.part_stream):
            raise StopIteration

        #remove any notes that have ended
        self.current = [note if note is not None and note.offset + note.duration > self.offset else None for note in self.current]

        #update the current notes based on the next notes stack in the part stream
        note_stack = self.part_stream[self.i]
        for note in note_stack:
            self.current[note.voice] = note

        #set the durations and syllables of the current notes to fit the shortest note
        delta = min([note.duration for note in self.current if note is not None])
        current_stack = []
        for note in self.current:
            if note is None:
                current_stack.append(None)
                continue
            attack = self.offset == note.offset
            release = note.offset + note.duration == self.offset + delta
            if note.phonemes:
                new_phonemes = StaticSyllable(attack=note.phonemes.attack if attack else '',  sustain=note.phonemes.sustain, release=note.phonemes.release if release else '')
            else:
                new_phonemes = None

            current_stack.append(StaticNote.fromstaticnote(note, offset=self.offset, duration=delta, phonemes=new_phonemes))


        self.offset += delta
        self.i += 1

        if not self.spread:
            return current_stack #return notes, chords, and rests
        else:
            return [note for notes in current_stack if notes is not None for note in notes.spread()] #return a list of only notes or rests (spread chords into notes)



def load_music(initial_directory=""):
    """Select a musicxml file and return the parsed music21 object"""
    app = QApplication(sys.argv)
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog #todo->look into making this be native
    filename, _ = QFileDialog.getOpenFileName(parent=None, caption="Open MusicXML Sheet Music", directory=initial_directory, filter="MusicXML File (*.xml *.mxl *.musicxml)", options=options)
    if filename:
        return music21.converter.parse(filename)
    else:
        raise Exception("You must select a music xml file")


def parse_music(score):
    """convert the music21 score object to a more convenient recipe format"""
    parts = get_parts_streams(get_voice_parts(score))
    metadata = score._getMetadata()
    
    #compute number of singers per (non-solo) section
    # min_singers = {part_name: min_singers_per_part(part) for part_name, part in parts.items()}
    recommended_singers_by_part = {part_name: recommended_singers_per_part(part) for part_name, part in parts.items()}
    singers_per_section = lcm([num_singers for part_name, num_singers in recommended_singers_by_part.items() if 'solo' not in part_name.lower()])    
    # singers_per_section = lcm([singers_per_section, 6]) #ensure that at least 6 singers are in a given section
    singers_per_solo = 1
    num_singers = {part_name: (singers_per_solo if 'solo' in part_name.lower() else singers_per_section) for part_name in parts.keys()}
    parsed_score = {
        "song_name": metadata.title if metadata.title else "untitled",                      #title of the piece of music
        "voice_parts": list(parts.keys()),                                                  #list the names of each voice part
        "num_singers": num_singers,
        "excerpts": {part_name: get_excerpts(score, part, num_singers[part_name]) for part_name, part in parts.items()}
    }

    return parsed_score


def get_voice_parts(score):
    """return a list of the voice parts in the song"""

    #TODO: select only parts that belong to a human voice, and not other instruments
    
    raw_names = [part.partName for part in score.parts]
    parts = {}
    
    for raw_name, part in zip(raw_names, score.parts):
        if not is_voice_part(raw_name):
            print(f'skipping non-voice part {raw_name}')
            continue

        if raw_names.count(raw_name) > 1:
            i = 1
            while raw_name + ' ' + int_to_roman(i) in parts:
                i += 1
            name = raw_name + ' ' + int_to_roman(i)
        else:
            name = raw_name
        assert name not in parts
        parts[name] = part

    return parts

def is_voice_part(part_name):
    """return whether or not the part name indicates a voice, rather than an instrument"""

    #TODO->this will fail for things like tenor saxophone
    # possible_part_names = ['soparno', 'alto', 'tenor', 'bass', 'baritone', '']
    # for possible_name in possible_part_names:
    #     if possible_name in part_name.lower()
    #         return True
    # return False

    return True

def int_to_roman(number): #TODO->convert this to music21 helper function
   """Convert an integer to Roman numerals. from: https://code.activestate.com/recipes/81611-roman-numerals/"""
   if type(number) != type(1):
      raise TypeError("expected integer, got %s" % type(number))
   if not 0 < number < 4000:
      raise ValueError("Argument must be between 1 and 3999")   #consider replacing this with just returning a string of the number if it is out of the range so that any number of parts is allowed
   ints = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)
   nums = ('M',  'CM', 'D', 'CD','C', 'XC','L','XL','X','IX','V','IV','I')
   numeral = ""
   for val, char in zip(ints, nums):
      count = int(number / val)
      numeral += char * count
      number -= val * count
   return numeral


def get_excerpts(score, part, num_singers):
    """return a list of excerpts that comprise the voice part. multiple excerpts indicate chords and/or multiple voices per line"""

    excerpts = []
    measures = [element for element in part if type(element) is music21.stream.Measure]
    
    for singer_num in range(num_singers):
        excerpt = []

        # dynamics = 'mf' #TODO->need to update how dynamics are computed... I think these should be passed separately

        for note_stack in  part.spread_iter():
            note_idx = singer_num % len(note_stack)
            excerpt.append(note_stack[note_idx])

        add_grace_duration(excerpt) #grace notes initiall have 0 duration. add a duration to them
        excerpts.append(excerpt)

    return excerpts


def get_parts_streams(parts):
    """attach phonemes to every note in the score"""

    parts_streams = {}
    
    for part_name, part in parts.items():
        part_stream, max_voice_splits = assemble_part_stream(part)
        parts_streams[part_name] = PartStream(part_stream, max_voice_splits) #store a reference to this part_stream under the part_name

        for voice_num in range(max_voice_splits):
            head = 0
            current_word = []
            while True: #(coordinates := get_next_note(part_stream, voice_num, head)) is not None:
                
                #collect the next word
                coordinates = get_next_note(part_stream, voice_num, head)
                if coordinates is None: 
                    assemble_word(current_word) #attach phonemes to any remeaining notes in current word
                    break
                note = get_note_at(part_stream, coordinates)
                
                if note.lyrics and note.lyrics.syllabic in ['single', 'begin']: #TODO->eventually allow multiple verses
                    assemble_word(current_word)
                    current_word = [] #reset for the next word

                current_word.append(note)

                head = coordinates[0] + 1

    return parts_streams

def assemble_part_stream(part):
    """convert the song to an easy to work with data structure for extracting lyrics"""
    
    measures = [element for element in part if type(element) is music21.stream.Measure]
    part_stream = []
    max_splits = 1
    for i, measure in enumerate(measures):
        measure_offset = measure.offset
        if music21.stream.Voice in [type(e) for e in measure]:
            voices = [voice for voice in measure if type(voice) is music21.stream.Voice]
        else:
            voices = [measure]

        if len(voices) > max_splits:
            max_splits = len(voices)
        
        notes = [
                    [
                        [
                            StaticNote.from21element(element, voice=voice_num, offset=frac(element.offset)+frac(measure_offset))
                        ] for element in voice if type(element) in [music21.note.Note, music21.note.Rest, music21.chord.Chord]
                    ] for voice_num, voice in enumerate(voices)
                ]

        # for voice_num, note_sequence in enumerate(notes):
        #     for note_stack in note_sequence:
        #         for note in note_stack:
        #             note.voice_num = voice_num #keep track of wich voice is singing the given note

        merged_measure = merge_measure(*notes)
        part_stream += merged_measure

    return part_stream, max_splits

def merge_measure(*voices):
    """recursively merge the lists of notes into a single merged list object"""
    
    num_voices = len(voices)
    if num_voices == 1:
        return voices[0]
    elif num_voices > 2:
        return merge_measure(
            merge_measure(*voices[:int(num_voices / 2)]), 
            merge_measure(*voices[int(num_voices / 2):])
        )

    assert(num_voices == 2)
    voice1, voice2 = voices
    head1, head2 = 0, 0

    merged_voice = []

    while head1 < len(voice1) and head2 < len(voice2):
        offset1, offset2 = voice1[head1][0].offset, voice2[head2][0].offset
        if offset1 < offset2:
            merged_voice.append(voice1[head1])
            head1 += 1
        elif offset2 < offset2:
            merged_voice.append(voice2[head2])
            head2 += 1
        else: #merge into stack
            merged_voice.append(voice1[head1] + voice2[head2])
            head1 += 1
            head2 += 1

    while head1 < len(voice1):
        merged_voice.append(voice1[head1])
        head1 += 1
    while head2 < len(voice2):
        merged_voice.append(voice2[head2])
        head2 += 1

    return merged_voice

def get_note_at(part_stream, coordinates):
    """return the note located in the part stream at the specified coordinates"""
    try:
        i, j = coordinates
        return part_stream[i][j]
    except:
        return None

def get_next_note(part_stream, voice_num, head):
    """return the coordinates of the next note in the part stream that belongs to this part"""
    
    for i, note_stack in enumerate(part_stream[head:]):
        for j, note in enumerate(note_stack):
            if note.voice == voice_num:
                return (head + i, j)

    return None

def assemble_word(word_elements):
    """construct the word from the list of notes, and attach the phonemes for that word"""
    
    #get a list of the notes without any rests
    word_notes = [note for note in word_elements if note.is_sung()]

    #if there are no sung notes or all sung notes have no lyrics, then return
    if len(word_notes) == 0 or sum(bool(note.lyrics) for note in word_notes) == 0:
        return

    word = ''
    for note in word_notes:
        if note.lyrics:
            word += note.lyrics.text

    #assign the constructed word to all notes that construct this word (TODO->this is probably unnecessary)
    for note in word_notes:
        note.word = word

    phonemes = get_phonetics(word)
    if phonemes is None: #TODO->if phonemes is None, means we need to look for substrings or continuations
        print(f'ERROR: "{word}" is not in the phonetic dictionary. replacing with default_phoneme "{default_phoneme}"')
        pdb.set_trace()
        phonemes = default_phoneme
    
    syllables = split_phonemes_into_syllables(phonemes)


    attacks = [bool(note.lyrics) for note in word_notes]
    releases = [(i+1 >= len(word_notes) or attacks[i+1]) for i in range(len(word_notes))]

    i = 0
    for note, attack, release in zip(word_notes, attacks, releases):
        if attack:
            cur_syllable = syllables[i]
            i += 1

        attack_phonemes = cur_syllable[0] if attack else ''
        sustain_phonemes = cur_syllable[1]
        release_phonemes = cur_syllable[2] if release else ''
        syllable = StaticSyllable(attack_phonemes, sustain_phonemes, release_phonemes)
        note.phonemes = syllable


    #for each note in word notes, 
    #determine if it has attack, and or release
    #attach syllables to each note (remove attack/release for notes without attach/releae)
    #first note, and notes following a released note, move to the next syllable
    #for the last note in the word notes, all remeaining phonemes get squished onto it? for now just truncate...

    # print(syllables)
    # print(phonemes, '->', word_notes)


def split_phonemes_into_syllables(phonemes):
    """return a list of syllables, i.e. (attack, sustain, release)"""

    raw_syllables = []
    raw_syllable = ''
    prev_was_consonant = True #if the last phoneme was a consonant
    
    i = 0
    while i < len(phonemes):
        p = phonemes_starts_with(diphthongs, phonemes, i)
        if p is not None:
            if prev_was_consonant:
                raw_syllable += p
            else:
                raw_syllables.append(raw_syllable)
                raw_syllable = '' + p
            prev_was_consonant = False
            i += len(p)
            continue
        
        p = phonemes_starts_with(consonants, phonemes, i)
        if p is not None:
            if prev_was_consonant:
                raw_syllable += p
            else:
                raw_syllables.append(raw_syllable)
                raw_syllable = '' + p
            prev_was_consonant = True
            i += len(p)
            continue

        p = phonemes_starts_with(vowels, phonemes, i)
        if p is not None:
            if prev_was_consonant:
                raw_syllable += p
            else:
                raw_syllables.append(raw_syllable)
                raw_syllable = '' + p
            prev_was_consonant = False
            i += len(p)
            continue

        pdb.set_trace()
        raise Exception(f'ERROR: phonetic word "{phonemes}" starting at "...{phonemes[i:]}" contains letters not in phonetic vowels, consonants or diphthongs')

    if len(raw_syllables) > 0 and prev_was_consonant:
        raw_syllables[-1] += raw_syllable
    else:
        raw_syllables.append(raw_syllable)

    #convert raw syllables to (attack, sustain, release) tuples
    syllables = [syllable_asr(raw_syllable) for raw_syllable in raw_syllables]

    return syllables

def phonemes_starts_with(phoneme_set, phonemes, i=0):
    """if phonemes starts with a phoneme in phoneme_set, return that phoneme, otherwise None. i indicates index to start from"""
    for p in phoneme_set:
        if phonemes.startswith(p, i):
            return p
    return None


def syllable_asr(raw_syllable):
    """convert the phoneme syllable into attack sustain release tuple"""

    #verify that the word is made of correct characters
    i = 0
    while i < len(raw_syllable):
        p = phonemes_starts_with(consonants + diphthongs + vowels, raw_syllable, i)
        assert(p is not None)
        i += len(p)

    #break the syllable into an [optional] initial consonant (attack), middle vowel (sustain), and [optional] ending consonant (release)
    attack, sustain, release = '', '', ''

    i = 0
    while i < len(raw_syllable):
        p = phonemes_starts_with(consonants, raw_syllable, i)
        if p is None:
            break
        attack += p
        i += len(p)

    while i < len(raw_syllable):
        p = phonemes_starts_with(diphthongs + vowels, raw_syllable, i)
        if p is None:
            break
        sustain += p
        i += len(p)

    while i < len(raw_syllable):
        p = phonemes_starts_with(consonants, raw_syllable, i)
        if p is None:
            break
        release += p
        i += len(p)

    #if sustain is diphthong, put second phoneme of diphthong on release
    p = phonemes_starts_with(diphthongs, sustain)
    if p == sustain and len(sustain) == 2:
        sustain = p[0]
        release = p[1] + release


    try:
        assert attack + sustain + release == raw_syllable   #we should have successfuly split the entire syllable
        assert len(sustain) > 0     #syllabars are of the form {C},V,{C}, i.e. sustain must contian a vowel
    except:
        pdb.set_trace()

    return (attack, sustain, release)


def attach_phonemes_to_single_element(element, phonemes):
    """attach the phonemes to the element. for chords, attach the phonemes to every sub note"""
    if type(element) is music21.note.Note:
        element.lyric = phonemes
    elif type(element) is music21.chord.Chord:
        # element.phonemes = phonemes
        for note in element:
            note.lyric = phonemes
    else:
        raise Exception(f'ERROR: unexpected type to attach phonemes to: {element}')

def is_word_in_dictionary(word):
    """returns whether a word is in the dictionary or not"""
    word = remove_punctuation(word)
    return word in phonetic_dictionary


def substring_is_in_phonetic_dictionationary(substring):
    """check if a substring is in the phonetic dictionary. if true, returns original word"""
    
    for word in phonetic_dictionary.keys():
        if word.startswith(substring):
            return word
    return None


def get_phonetics(word):    
    """return the IPA phonemes that make the given word (TBD how to handle homographs)"""
    word = remove_punctuation(word)
    try:
        return phonetic_dictionary[word]
    except:
        return None

def remove_punctuation(word):
    """return a (lowercase) string without any punctuation"""
    if word is None: word = ''
    word = [char.lower() for char in word if char.lower() in "abcdefghijklmnopqrstuvwxyz'"] #replace this with the character set for the language. apostrophe included for contractions, e.g. I'll, fav'rite, etc.
    word = (''.join(word)).lower()
    return word



def get_max_voice_split(part):
    """return the maximum number of voices a part splits into"""
    max_split = 0
    measures = [element for element in part if type(element) is music21.stream.Measure]
    for measure in measures:
        if music21.stream.Voice in [type(e) for e in measure]:
            voices = [voice for voice in measure if type(voice) is music21.stream.Voice]
        else:
            voices = [measure]

        if len(voices) > max_split:
            max_split = len(voice)

    return max_split



def add_grace_duration(excerpt):
    """increase duration of grace notes from 0 to a small fraction of the proceeding note"""
    #WARNING: FOR NOW, ONLY SINGLE GRACE NOTES AT A TIME ARE HANDLED
    for i, note in enumerate(excerpt):
        if note['duration'] == 0: #TODO->add a tag that indicates a grace note + grace duration
            grace = excerpt[i]
            normal = excerpt[i+1] #TODO->add some sort of check for if the grace note is the last note
            grace_duration = min(0.1, normal['duration'] * 0.2)
            grace['duration'] = grace_duration
            normal['duration'] -= grace_duration




volume_map = { #map from volume name to an amplitude scale. this should probably be adjusted/nonlinear/etc.
    'ppp':  0.125, 
    'pp':   0.25,
    'p':    0.375,
    'mp':   0.50,
    'mf':   0.675,
    'f':    0.75,
    'ff':   0.875,
    'fff':  1.0,
    'fp':   0.675, #TODO->adjust these. eventually dynamcs should be contour based, note strictly hardcoded values
    'sfz':  1.0,
}


def get_measure_notes(voice, chord_num, state):
    """get the notes/rests from a measure (single voice only) with the specified chord number"""
    notes = []

    for element in voice:
        if type(element) is music21.chord.Chord: #split this voice off to a specific note in the chord based on their number
            element = element[chord_num % len(element)]
        
        if type(element) is music21.note.Note:
            duration = get_note_duration(state=state, note=element)
            notes.append({
                'volume': volume_map[state.dynamics],
                'duration': duration, #60 / tempo * element.quarterLength,
                'pitch': element.pitch.frequency,
                #todo->get the correct syllable
                # 'syllable': element.lyric, #custom property attached to all notes
            })
            state.beat += duration

        elif type(element) is music21.note.Rest:
            duration = get_note_duration(state=state, note=element)
            notes.append({
                'volume': 0.0,
                'duration': duration, #60 / tempo * element.quarterLength
            })
            state.beat += duration

        elif type(element) is music21.dynamics.Dynamic:
            state.dynamics = element.value

        #else skip all others?            

    return notes


def get_tempo(state):
    """Return the tempo at the specified beat of the song"""
    for start, stop, tempo in state.metronome:
        if state.beat >= start and state.beat <= stop:
            return tempo.getQuarterBPM()

def get_note_duration(state, note):
    """compute the total duration of a note based on all temp changes"""
    #TODO->implement correctly
    
    return frac(note.quarterLength)


def get_singer_voice_splits(part):
    """determine the paths that individual singers would take through the part based on chords and multiple voicings"""
    measures = [element for element in part if type(element) is music21.stream.Measure]
    singer_voice_splits = []

    for measure in measures:
        if music21.stream.Voice in [type(e) for e in measure]:
            splits = [singers_per_voice(voice) for voice in measure if type(voice) is music21.stream.Voice]
        else:
            splits = [singers_per_voice(measure)]

        singer_voice_splits.append(splits)

    return singer_voice_splits

def singers_per_voice(measure):
    """count the number of singers in the measure provided. the measure can contain chords, but must be a single voice"""
    num_singers = 1

    for element in measure:
        # if type(element) is music21.note.Note:      #single note is sung by a single person
        #     if num_singers < 1:
        #         num_singers = 1
        if type(element) is music21.chord.Chord:  #chord is sung by number of notes in chord
            if num_singers < len(element):
                num_singers = len(element)

    return num_singers


def min_singers_per_part(part):
    """compute the minimum singers needed for a given voice part"""
    singer_voice_splits = get_singer_voice_splits(part)
    min_singers = max([sum(measure) for measure in singer_voice_splits])
    return min_singers


def recommended_singers_per_part(part):
    """compute the recommended number of singers per voice part so that notes in chords have the same number of singers per note"""
    return lcm([len(stack) for stack in part.spread_iter()])

    # singer_voice_splits = get_singer_voice_splits(part)
    # recommended_singers = lcm([sum(measure) for measure in singer_voice_splits])
    # return recommended_singers


def lcm(factors):
    """least common multiple"""
    factors = [f for f in factors if f != 0] #filter out zeros in factors
    
    if len(factors) == 0:
        return 0

    l = factors[0]
    for f in factors[1:]:
        if f == 0: continue
        l = int(l*f/gcd(l,f))
    return l

def sum_samples(s1, s2):
    if s1.shape[0] > s2.shape[0]: #make shorter sample come first
        s1, s2 = s2, s1
    difference = s2.shape[0] - s1.shape[0]
    return s2 + np.pad(array=s1, pad_width=(0, difference), mode='constant')    



if __name__ == '__main__':
    from singer import singer, play

    raw_score = load_music("music")
    print('Parsing score...', end='')
    sys.stdout.flush()
    parsed_score = parse_music(raw_score)
    print('Done')
    
    FS_out = 44100
    matt = singer(singer_name='matt', FS_out=FS_out)

    print('Creating samples...')
    sys.stdout.flush()
    ensemble_output = None
    num_singers = sum([n for n in parsed_score['num_singers'].values()])
    for part_name, split_parts in parsed_score['excerpts'].items():
        section_output = None
        for i, part in enumerate(split_parts):
            print(f'{part_name} #{i}')
            sample = matt.sing_excerpt(part)
            if section_output is None:
                section_output = sample
            else:
                section_output = sum_samples(section_output, sample)
        
        section_output /= parsed_score['num_singers'][part_name]
        if ensemble_output is None:
            ensemble_output = section_output
        else:
            ensemble_output = sum_samples(ensemble_output, section_output)

    # ensemble_output /= len(parsed_score['num_singers']) #divide by number of sections
    ensemble_output /= num_singers
    ensemble_output = np.tanh(ensemble_output) #squash output so no samples are greater than +/-1

    #add reverb. TODO->ensure that this is a float, not 16-bit (have function load the thing)
    print('Adding Reverb')
    _, reverb_IR = wavfile.read('reverb/reverb1.wav')
    # ensemble_output = np.convolve(ensemble_output, reverb_IR)


    print('Done')

    print('Playing output')
    sys.stdout.flush()
    play(ensemble_output, FS_out, block=False)


    pdb.set_trace()
    
    wavfile.write(f"output/{parsed_score['song_name']}.wav", FS_out, ensemble_output)
    wavfile.write(f"output/{parsed_score['song_name']}_reverb.wav", FS_out, np.convolve(ensemble_output, reverb_IR))
    pdb.set_trace()