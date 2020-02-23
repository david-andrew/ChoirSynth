"""
Notes

videos of music staff
videos with a single instrument at a time



will need to manually screen each video?

Keywords
[instrument]
etudes
beginner etude[s]
intermediate etude[s]
advanced etude[s]
study #
for solo [instrument]
lmea all-state

solo performance


examples:
[instrument] etude 2019


Channels:
Nathos Ost Music
"""

import pdb

import youtube_dl as ydl




brass = ['trumpet', 'b flat trumpet', 'f trumpet', 'piccolo trumpet', 'd trumpet', 'c trumpet', 'french horn', 'euphonium', 'trombone', 'bass trombone', 'tenor trombone', 'alto trombone', 'baritone horn', 'tuba']
woodwinds = ['violin', 'viola', 'cello', 'double bass', 'violoncello', 'contrabass']
woodwind = ['flute', 'piccolo', 'clarinet', 'b flat clarinet', 'saxophone', 'oboe', 'bassoon', 'english horn', 'e flat clarinet', 'alto clarinet', 'bass clarinet', 'contrabassoon']
misc = ['guitar']


shortlist = ['flute', 'oboe', 'clarinet', 'bassoon', 'french horn' 'horn in f', 'trumpet', 'trombone', 'tuba']

search_functions = [
    lambda i: f"{i} etude",
    lambda i: f"beginner {i} etude",
    lambda i: f"intermediate {i} etude",
    lambda i: f"advabced {i} etude",
    lambda i: f"{i} study #",
    lambda i: f"for solo {i}",
    lambda i: f"{i} solo performance",
    lambda i: f"{i} etude 2019",
    lambda i: f"{i} etude 2018",
    lambda i: f"{i} etude 2017",
    lambda i: f"{i} lmea all-state",
    lambda i: f"{i} etude",
    lambda i: f"{i} etude",
    lambda i: f"{i} etude",
    lambda i: f"{i} etude",
    lambda i: f"{i} etude",
    
    ]


for instrument in shortlist:
    for searc_function in search_functions:
        search_term = search_function(instrument)
        pdb.set_trace()
