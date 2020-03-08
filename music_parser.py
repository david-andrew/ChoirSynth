import pdb

import os
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
import music21
from math import gcd
import numpy as np
from fractions import Fraction as frac
# from numpy import lcm
import json


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

class StaticNote():
    def __init__(self, voice=None, pitch=None, duration=None, offset=None, word=None, phonemes=None, lyrics=None):
        self.voice = voice
        self.pitch = pitch #can be either a single number, or array of pitches to represent a chord. None indicates rest
        self.duration = duration
        self.offset = offset
        self.word = word
        self.phonemes = phonemes
        self.lyrics = lyrics

    def __getitem__(self, key):
        if key == 'voice':
            return self.voice
        elif key == 'pitch':
            return self.pitch
        elif key == 'duration':
            return self.duration
        elif key == 'offset':
            return self.offset
        elif key == 'word':
            return self.word
        elif key == 'phonemes':
            return self.phonemes
        elif key == 'lyrics':
            return self.lyrics
        else:
            raise Exception(f'ERROR: unrecognized key "{key}" for StaticNote')

    def keys(self):
        return ['voice', 'pitch', 'duration', 'offset', 'word', 'phonemes', 'lyrics']

    def __repr__(self):
        word = '' if self.word is None else f', word: \'{self.word}\''
        phonemes = '' if self.phonemes is None else f', phonemes: \'{self.phonemes}\''
        lyrics = '' if self.lyrics is None else f', lyrics: {self.lyrics}'
        if self.pitch is None:
            return f'<Rest voice: {self.voice}, duration: {self.duration}, offset: {self.offset}>'
        elif isinstance(self.pitch, list):
            return f'<Chord voice: {self.voice}, pitches: {self.pitch}, duration: {self.duration}, offset: {self.offset}{word}{phonemes}{lyrics}>'
        else:
            return f'<Note voice: {self.voice}, pitch: {self.pitch}, duration: {self.duration}, offset: {self.offset}{word}{phonemes}{lyrics}>'

    def __str__(self):
        #TODO->make this a more compact representation
        return repr(self)

    def __hash__(self):
        return hash(self.voice, self.pitch, self.duration, self.offset, self.word, self.phonemes, self.lyrics)
       
    def __eq__(self, other):
        return (self.voice == other.voice
            and self.pitch == other.pitch 
            and self.duration == other.duration 
            and self.offset == other.offset 
            and self.word == other.word
            and self.phonemes == other.phonemes
            and self.lyrics == other.lyrics)

    @staticmethod
    def from21element(element, voice=None, **kwargs):
        """convert the music21 element to a static note"""
        duration = frac(element.quarterLength)
        offset = frac(element.offset)

        if type(element) is music21.chord.Chord:
            pitch = [note.pitch.frequency for note in element]
            lyrics = StaticNoteLyrics.fromlyrics(element.lyrics)
        elif type(element) is music21.note.Note:
            pitch = element.pitch.frequency
            lyrics = StaticNoteLyrics.fromlyrics(element.lyrics)
        elif type(element) is music21.note.Rest:
            pitch = None
            lyrics = None
        else:
            raise Exception(f'ERROR: unknown music21 element type {element}')

        raw_static_m21 = StaticNote(voice=voice, pitch=pitch, duration=duration, offset=offset, lyrics=lyrics)
        return StaticNote(**dict(raw_static_m21, **kwargs))

    @staticmethod
    def fromstaticnote(static_note, **kwargs):
        """return a new StaticNote given an old one plus keyword arguments"""
        return StaticNote(**dict(static_note, **kwargs))

class StaticNoteLyrics():
    def __init__(self, text=None, syllabic=None):
        self.text = text
        self.syllabic = syllabic

    def __str__(self):
        if (self):
            return f'<Lyrics text: \'{self.text}\', syllabic: {self.syllabic}>'
        else:
            return 'None'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.text == other.text and self.syllabic == other.syllabic

    def __hash__(self):
        return hash(self.text, self.syllabic)

    def __bool__(self):
        return self.text is not None and self.syllabic is not None

    @staticmethod
    def fromlyrics(lyrics):
        """for now only a single line of lyrics are allowed"""
        if lyrics:
            return StaticNoteLyrics(text=lyrics[0].text, syllabic=lyrics[0].syllabic)
        else:
            return StaticNoteLyrics()

class StaticSyllable():
    def __init__(self, attack=None, sustain=None, release=None):
        if attack is None:
            attack = ''
        if sustain is None:
            sustain = ''
        if release is None:
            release = ''

        self.attack = attack
        self.sustain = sustain
        self.release = release

    def __str__(self):
        return f'{self.attack}{self.sustain}{self.release}'

    def __repr__(self):
        return f'attack: {self.attack}, sustain: {self.sustain}, release: {self.release}'

    def __eq__(self, other):
        return self.attack == other.attack and self.sustain == other.sustain and self.release == other.release

    def __hash__(self):
        return hash(self.attack, self.sustain, self.release)

    def __bool__(self):
        return str(self) != ''



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
    parts = get_voice_parts(score)
    metadata = score._getMetadata()

    attach_lyrics_to_parts(parts)
    
    #compute number of singers per (non-solo) section
    # min_singers = {part_name: min_singers_per_part(part) for part_name, part in parts.items()}
    recommended_singers = {part_name: recommended_singers_per_part(part) for part_name, part in parts.items()}
    singers_per_section = lcm([num_singers for part_name, num_singers in recommended_singers.items() if 'solo' not in part_name.lower()])    
    singers_per_section = lcm([singers_per_section, 6]) #ensure that at least 6 singers are in a given section
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

def int_to_roman(number):
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


def get_excerpts(score, part, num_singers=None):
    """return a list of excerpts that comprise the voice part. multiple excerpts indicate chords and/or multiple voices per line"""

    if num_singers is None:
        num_singers = max(min_singers_per_part(part), 6) #ensure at least 6 singers per part, if specific number not specified

    excerpts = []
    measures = [element for element in part if type(element) is music21.stream.Measure]
    splits = get_singer_voice_splits(part)



    state = type('test', (), {})()                      #empty container class to hold the current state of the singer
    state.score = score                                 #store the score in the object
    state.metronome = score.metronomeMarkBoundaries()   #used for calculating tempo
    # landmarks = get_measure_landmarks(state, part) #compute the beat # for the start of each measure


    for singer_num in range(num_singers): #n'th singer run through the voice part
        excerpt = []
        
        state.dynamics = 'mf'                               #reset dynamics to default
        state.beat = 0                                      #reset beat count to zero
       
        for measure, split in zip(measures, splits):
            if len(split) == 1:
                voice = measure
                chord_num = singer_num % split[0]
            else:
                remainder = singer_num % sum(split)
                voice_num = 0
                while remainder > sum(split[:voice_num+1]):
                    voice_num += 1
                # measure = [element for element in measure if type(element) is music21.]
                voices = [voice for voice in measure if type(voice) is music21.stream.Voice]
                voice = voices[voice_num]
                chord_num = remainder - sum(split[:voice_num])
            
            excerpt += get_measure_notes(voice, chord_num, state)
        
        #final post processing of the excerpt
        add_grace_duration(excerpt) #grace notes initiall have 0 duration. add a duration to them
        excerpts.append(excerpt)
    
    return excerpts

#this is literally measure.offset!
# def get_measure_landmarks(state, part):
#     """compute the beat at the start of each measure. Used to ensure choir remians in time"""
    
#     pdb.set_trace()

def attach_lyrics_to_parts(parts):
    """attach phonemes to every note in the score"""

    # #attach None to every note's phonemes attribute
    # for part in parts.values():
    #     for element in part:
    #         if type(element) in [music21.note.Note, music21.chord.Chord]:
    #             attach_phonemes_to_single_element(element, None)
    #             # element.phonemes = None



    for part_name, part in parts.items():
        part_stream, max_splits = assemble_part_stream(part)
        
        for voice_num in range(1): #range(max_splits): #for now just use voice 1
            head = 0
            current_word = []
            while True: #(coordinates := get_next_note(part_stream, voice_num, head)) is not None:
                
                #collect the next word
                coordinates = get_next_note(part_stream, voice_num, head)
                if coordinates is None: 
                    break
                note = get_note_at(part_stream, coordinates)
                
                if note.lyrics and note.lyrics.syllabic in ['single', 'begin']: #TODO->eventually allow multiple verses
                    assemble_word(current_word)
                    current_word = [] #reset for the next word

                current_word.append(note)

                head = coordinates[0] + 1

        for note in part_stream: 
            print(note)
        pdb.set_trace()

    pdb.set_trace() #algorithm to collect words and then attach phonemes


    # #attach the default phonemes to any notes that didn't get phonemes
    # for part in parts.values():
    #     for element in part:
    #         if type(element) in [music21.note.Note, music21.chord.Chord] and element.lyric is None:
    #             attach_phonemes_to_single_element(element, default_phoneme)


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
    word_notes = [note for note in word_elements if type(note) is not music21.note.Rest]

    if len(word_notes) == 0:
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
    singer_voice_splits = get_singer_voice_splits(part)
    recommended_singers = lcm([sum(measure) for measure in singer_voice_splits])
    return recommended_singers


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

    # print('Creating samples...', end='')
    print('Creating samples...')
    sys.stdout.flush()
    ensemble_output = None
    section_output = None
    num_singers = sum([n for n in parsed_score['num_singers'].values()])
    for part_name, split_parts in parsed_score['excerpts'].items():
        print(f'--> {part_name}')
        for part in split_parts:
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
    ensemble_output /= len(parsed_score['num_singers']) #divide by number of sections
    print('Done')

    print('Playing output')
    sys.stdout.flush()
    play(ensemble_output, FS_out, block=False)
    
    pdb.set_trace()
    # from scipy.io import wavfile
    # wavfile.write(f'output/{parsed_score['song_name']}.wav', FS_out, ensemble_output)