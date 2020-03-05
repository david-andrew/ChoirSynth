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

with open('phonetic_dictionary.json') as f:
    phonetic_dictionary = json.load(f)

default_phoneme = 'u'

class LyricDAG():
    def __init__(self, coordinates, parents=None, children=None):
        self.parents = parents if parents is not None else []
        self.children = children if children is not None else []
        self.coordinates = coordinates


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

    attach_phonemes(parts)
    
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

def attach_phonemes(parts):
    """attach phonemes to every note in the score"""

    #attach None to every note's phonemes attribute
    for part in parts.values():
        for element in part.flat:
            if type(element) is music21.note.Note:
                element.phonemes = None



    for part_name, part in parts.items():
        part_lyrics = assemble_lyrics(part)
        
        pdb.set_trace() #algorithm to collect words and then attach phonemes

        # for coordinates in offsets:
        #     attach_single_word(lyrics, offsets, coordinates)
            

        pdb.set_trace()
        



    #construct arrays for notes for every voice in the voice parts

    #attach the phonemes for the lyrics to the notes
    # for part_name, part in parts.items():
    #     for measure in [element for element in part if type(element) is music21.stream.Measure]:
    #         if music21.stream.Voice in [type(e) for e in measure]:
    #             voices = [voice for voice in measure if type(voice) is music21.stream.Voice]
    #         else:
    #             voices = [measure]
            
    #         for voice in voices:
    #             for element in voice:
    #                 if type(element) is music21.note.Note and element.phonemes is None:
    #                     attach_single_word(element)
    # for part in parts.values():
    #     for element in part.flat:
    #         if type(element) is music21.note.Note and element.phonemes is None:
    #             attach_single_word(element)


    #attach the default phonemes to any notes that didn't get phonemes
    for part in parts.values():
        for element in part.flat:
            if type(element) is music21.note.Note and element.phonemes is None:
                element.phonemes = default_phoneme


# def assemble_lyrics(part):
#     """extracts lyrics maps/other useful structures for the specific voice part"""
#     lyrics = []
#     # pointers = {}
#     offsets = {}
    
#     measures = [element for element in part if type(element) is music21.stream.Measure]
#     for i, measure in enumerate(measures):
#         measure_lyrics = []
#         measure_offset = measure.offset
#         if music21.stream.Voice in [type(e) for e in measure]:
#             voices = [voice for voice in measure if type(voice) is music21.stream.Voice]
#         else:
#             voices = [measure]
        
#         for j, voice in enumerate(voices):
#             voice_lyrics = []

#             notes = [element for element in voice if type(element) in [music21.note.Note, music21.chord.Chord]]

#             for k, note in enumerate(notes):                
#                 voice_lyrics.append(note)
#                 coordinates = (i,j,k) # (measure, voice, note)
#                 # pointers[coordinates] = note
#                 offsets[coordinates] = measure_offset + note.offset

#             measure_lyrics.append(voice_lyrics)
#         lyrics.append(measure_lyrics)

#     return lyrics, offsets

def assemble_lyrics(part):
    """convert the song to an easy to work with data structure for extracting lyrics"""
    
    measures = [element for element in part if type(element) is music21.stream.Measure]
    stream = []
    for i, measure in enumerate(measures):
        measure_offset = measure.offset
        if music21.stream.Voice in [type(e) for e in measure]:
            voices = [voice for voice in measure if type(voice) is music21.stream.Voice]
        else:
            voices = [measure]

        
        notes = [[[element] for element in voice if type(element) in [music21.note.Note, music21.chord.Chord]] for voice in voices]
        for note_stacks in notes:
            for note_stack in note_stacks:
                for note in note_stack:
                    note.offset += measure_offset
        merged_measure = merge_measure(*notes)
        stream += merged_measure

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



# def attach_single_word(lyrics, offsets, coordinates):
#     """attach the phonemes for a single word"""
#     i, j, k = coordinates
#     element = lyrics[i][j][k]
#     assert(type(element) in [music21.note.Note, music21.chord.Chord])

#     #for now, don't worry about notes vs chords. instead just construct the word. When we attach the phoneme, we'll have a function handle chords vs notes for us
#     assert(element.lyrics) #assert lyrics arent empty
#     assert(element.lyrics[0].syllabic in ['begin', 'single'])


#     # #construct the whole word
#     # word = ''

#     #entry point to collect all notes for this word
#     init_node = LyricDAG(coordinates=coordinates)

#     #keep track of the roots of the DAG that we collect for this word
#     root_nodes = []
#     root_nodes.append(init_node)


#     #DFS on the note until we collected the whole note
#     dfs_stack = [] #.append() and .pop() for stack operations
#     visited = set()
#     dfs_stack.append(init_node)

#     while (len(dfs_stack) > 0):
#         curr_node = dfs_stack.pop()
#         visited.add(curr_node.coordinates)

#         prev_coordinates = get_prev_notes(lyrics, offsets, curr_node.coordinates)
#         for coordinates in prev_coordinates:
#             if coordinates not in visited:
#                 node = LyricDAG(coordinates=coordinates)
#                 curr_node.parents.append(node) #build out the DAG
#                 dfs_stack.append(node) #add this node to be expanded next

#         next_coordinates = get_next_notes(lyrics, offsets, curr_node.coordinates)
#         for coordinates in next_coordinates:
#             if coordinates not in visited:
#                 node = LyricDAG(coordinates=coordinates)
#                 curr_node.children.append(node)
#                 dfs_stack.append(node)




#         pdb.set_trace()

#     pdb.set_trace()


#     # #if single, check for sustain in the next note/voices
#     # if element.lyrics[0].syllabic == 'single':
#     #     word += element.lyrics[0].text
#     #     coordinates = get_next_notes(lyrics, coordinates)
#     #     if not is_lyrics_sustained(lyrics, offsets, coordinates):
#     #         #whole word is on this single note
#     #         print(word)
#     #         #convert word to phonetics
#     #         phonemes = get_phonetics(word)
#     #         attach_phonemes_to_single_element(element, phonemes)
#     #     else:
#     #         pdb.set_trace()
#     # else: #begin
#     #     while True:
#     #         word += element.lyrics[0].text
#     #         coordinates = get_next_notes(lyrics, coordinates)
#     #         i, j, k = coordinates
#     #         element = lyrics[i][j][k]
#     #         #check if element is not last note in word
#     #         #if element contains syllable, include its text
#     #         #etc.
#     #         pdb.set_trace()
#     #     pdb.set_trace()



#     # if type(element) is music21.note.Note:
#     #     note = element
#     #     if note.phonemes is None:
#     #         pdb.set_trace()
#     #         # assert(element.lyrics)
#     #         #assert that this is a beginning note. otherwise it should already have phonemes
#     # else:
#     #     #treat as a single note, but attach phonemes to every 
#     #     pdb.set_trace()


#     pdb.set_trace()


def attach_phonemes_to_single_element(element, phonemes):
    """attach the phonemes to the element. for chords, attach the phonemes to every sub note"""
    if type(element) is music21.note.Note:
        element.phonemes = phonemes
    elif type(element) is music21.chord.Chord:
        for note in element:
            note.phonemes = phonemes
    else:
        raise Exception(f'ERROR: unexpected type to attach phonemes to: {element}')

# def is_lyrics_sustained(lyrics, offsets, coordinates):
#     """determine if the note at the given coordinates is sustained (empty could mean get lyrics from different voice)"""
#     i, j, k = coordinates
#     note = lyrics[i][j][k]
    
#     #if this note has lyrics, then it isn't a sustain from the previous
#     if note.lyrics != []:
#         return False

#     #check if any of the other voice parts have lyrics at this spot
#     pdb.set_trace()

# def get_next_notes(lyrics, offsets, coordinates):
#     """return an array of coordinates of the next notes in the part if any"""
#     pdb.set_trace()

#     #needs to determine if the next note is a part of this current note.
#     #probably need to pass single vs begin vs middle vs end...


#     i, j, k = coordinates
#     if k + 1 < len(lyrics[i][j]):
#         candidate_coordinates = (i, j, k + 1)
#         if is_lyrics_sustained(lyrics, offsets, candidate_coordinates):
#             return candidate_coordinates

#     #else extend into next measure, potentially multuple voices


#     pdb.set_trace()
#     pass

# def get_prev_notes(lyrics, offsets, coordinates):
#     """return an array of coordinates of the previous notes in the part if any"""
#     i, j, k = coordinates
#     element = lyrics[i][j][k]
    
#     #no previous notes for current note if current is the start or a whole word
#     if element.lyrics and element.lyrics[0].syllabic in ['begin', 'single']:
#         return []

#     pdb.set_trace()
#     pass

def get_phonetics(word):
    """return the IPA phonemes that make the given word (TBD how to handle homographs)"""
    word = remove_punctuation(word)
    try:
        phonemes = phonetic_dictionary['english']['u.s.'][word]
    except:
        phonemes = default_phoneme
    return phonemes

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
                'syllable': element.phonemes, #custom property attached to all notes
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