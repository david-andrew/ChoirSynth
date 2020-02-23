import pdb

import sys
import mido
import pyaudio
import time
import numpy as np
import multiprocessing


#todo: organ vs piano playback (i.e. const volume, no decay, or velocity-volume with decay)


#set up a shared memory location for the piano roll
multiprocessing_manager = multiprocessing.Manager()
roll = multiprocessing_manager.list([0] * 128)

pa = pyaudio.PyAudio()
pdb.set_trace()

def main():



    #set up input midi instrument
    ports = mido.get_input_names()
    with mido.open_input(ports[0]) as port:
        #set up callback function to recieve new midi data
        port.callback = recieve_message

        #set up audio output
        pa = pyaudio.PyAudio()
        stream = pa.open(rate=16000, channels=1, format=pyaudio.paFloat32, output=True, stream_callback=stream_audio)
        stream.start_stream()

        try:
            while True:             #loop forever. on ctrl-c call stream/pyaudio cleanup
                time.sleep(1)
        finally:
            print('Cleaning up audio objects...', end='')
            sys.stdout.flush()
            stream.stop_stream()
            stream.close()
            pa.terminate()
            print('Done')






def recieve_message(msg):
    """callback to update the piano roll based on the midi message received"""
    if msg.type in ['note_on', 'note_off']:
        roll[msg.note] = msg.velocity
    else: #do nothing for non-note on/off messages
        pass

    #DEBUG print out notes in piano roll
    for i in range(16):
        print(roll[8*i:8*(i+1)])
    print('')


def stream_audio(in_data, frame_count, time_info, status):
    """callback to write audio data to the sound device"""
    # np_roll = np.array(roll, dtype=np.float32)
    return np.zeros(frame_count, dtype=np.float32)


if __name__ == '__main__':
    main()