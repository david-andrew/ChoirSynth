from fractions import Fraction as frac
import music21

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


