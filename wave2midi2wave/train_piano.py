from zipfile import ZipFile
import json
from os.path import expanduser, join, split, splitext
from io import BytesIO
from scipy.io import wavfile
import mido
import pdb


data_path = join(expanduser('~'), 'Downloads', 'maestro-v2.0.0.zip') #'/home/David/Downloads/maestro-v2.0.0.zip' 
with ZipFile(data_path, 'r') as archive:
    
    #get the list of files in the archive.
    info = json.loads(archive.read(join('maestro-v2.0.0', 'maestro-v2.0.0.json')))
        
    #split into test, train, and validation sets
    test = list(filter(lambda entry: entry['split'] == 'test', info))
    train = list(filter(lambda entry: entry['split'] == 'train', info))
    validation = list(filter(lambda entry: entry['split'] == 'validation', info))

    #
    wav = BytesIO(archive.read(join('maestro-v2.0.0', test[0]['audio_filename'])))
    midi = archive.read(join('maestro-v2.0.0', test[0]['midi_filename']))

    wav = wavfile.read(wav)
    pdb.set_trace()
    midi_parser = mido.Parser()
    midi_parser.feed(midi)
    #for m in midi_parser: m
    #write function/library to convert from midi-messages to piano-roll data


    pdb.set_trace()
    pass