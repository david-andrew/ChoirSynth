#convert the arpabet dictionary to IPA

import pdb
import json

in_path = 'librispeech-lexicon.txt'
out_path = 'english_ipa_dictionary.txt'

letters = { #map from arpabet phoneme letters to ipa phoneme letters
    'AA' : 'ɑ',
    'AE' : 'æ',
    'AH' : 'ʌ',
    'AO' : 'ɔ',
    'AW' : 'aʊ',
    'AX' : 'ə',
    'AXR': 'ɚ',
    'AY' : 'aɪ',
    'EH' : 'ɛ',
    'ER' : 'ɝ',
    'EY' : 'eɪ',
    'IH' : 'ɪ',
    'IX' : 'ɨ',
    'IY' : 'i',
    'OW' : 'oʊ',
    'OY' : 'ɔɪ',
    'UH' : 'ʊ',
    'UW' : 'u',
    'UX' : 'ʉ',
    'B'  : 'b',
    'CH' : 'tʃ',
    'D'  : 'd',
    'DH' : 'ð',
    'DX' : 'ɾ',
    'EL' : 'l̩',
    'EM' : 'm̩',
    'EN' : 'n̩',
    'F'  : 'f',
    'G'  : 'ɡ',
    'HH' : 'h',
    'JH' : 'dʒ',
    'K'  : 'k',
    'L'  : 'l',
    'M'  : 'm',
    'N'  : 'n',
    'NG' : 'ŋ',
    'NX' : 'ɾ̃',
    'P'  : 'p',
    'Q'  : 'ʔ',
    'R'  : 'ɹ',
    'S'  : 's',
    'SH' : 'ʃ',
    'T'  : 't',
    'TH' : 'θ',
    'V'  : 'v',
    'W'  : 'w',
    'WH' : 'ʍ',
    'Y'  : 'j',
    'Z'  : 'z',
    'ZH' : 'ʒ',
}

def strip_digits(string):
    return ''.join(c for c in string if not c.isdigit())

output = '' #to store the ipa version of the dicitonary

with open(in_path) as f: #read in arpabet dictionary, parse, and convert to ipa
    for line in f:
        parse = [strip_digits(text) for text in line.split()]
        word = parse[0].lower()
        ipa = ''.join(letters[p] for p in parse[1:])
        output += "%s %s\n" % (word, ipa)

with open(out_path, 'w') as f: #save ipa to file
    f.write(output)