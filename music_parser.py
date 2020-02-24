import pdb

import os
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
import music21
from math import gcd
import numpy as np
from fractions import Fraction as frac
# from numpy import lcm


#parse musicxml sheet music and convert to individual excerpts for each singer
#TODO:
# for now we're skipping chords, as well as multiple voices on a line


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
    pass
    parts = get_voice_parts(score)
    metadata = score._getMetadata()
    
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
        # "excerpts": {part_name: get_excerpts(part) for part_name, part in parts.items()}    #map from voice part to its excerpt

        # "singers_per_part": {part._partName: singers_per_part(part) for part in score.parts},     #list the minimum singers needed for the given voice part
    }

    return parsed_score


def get_voice_parts(score):
    """return a list of the voice parts in the song"""
    
    raw_names = [part.partName for part in score.parts]
    parts = {}
    
    for raw_name, part in zip(raw_names, score.parts):
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

def get_measure_landmarks(state, part):
    """compute the beat at the start of each measure. Used to ensure choir remians in time"""
    
    pdb.set_trace()

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
                'syllable': 'a'
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
    
    FS_out = 192000
    matt = singer(singer_name='matt', FS_out=FS_out)

    print('Creating samples...', end='')
    sys.stdout.flush()
    ensemble_output = None
    section_output = None
    num_singers = sum([n for n in parsed_score['num_singers'].values()])
    for part_name, split_parts in parsed_score['excerpts'].items():
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
    # wavfile.write('out.wav', FS_out, ensemble_output)