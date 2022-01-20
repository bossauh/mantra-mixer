import os
import threading
import time
import math
from pathlib import Path
from queue import Empty, Queue
from typing import List, Union

import ffmpy
import numpy as np
import sounddevice as sd
import soundfile as sf

from .exceptions import *

RATE = 48000


class Track:
    """
    Initialize a Track.
    A track is what's responsible for opening a constant sd.Stream in the background.
    We can then play audio by putting ndarray data into this track's Queue.

    Notes
    -----
    - The shape of the ndarray must be the same as self.shape

    Parameters
    ----------
    `name` : str
        The name of this track.
    `vol` : int
        The initial volume of the track. Defaults to 1 which is 100%. You can go higher than 1 but it starts to sound shit.
    """

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name

        self.occupied = False
        self.shape = None
        self.samplerate = RATE

        self.queue = Queue()
        self.start()

        self.stopped = False
        self._stop_signal = False

        # Audio related attributes
        self.vol = kwargs.get("vol", 1)

        # Wait for the track to start
        while self.shape is None:
            time.sleep(0.01)

    def update_samplerate(self, rate: int) -> None:
        """
        Update the samplerate of this track.
        It does this by stopping the stream and changing the samplerate attribute and starting the stream again.

        Parameters
        ----------
        `rate` : int
            The samplerate.
        """

        if rate == self.samplerate:
            return

        self.stop()
        self.samplerate = rate
        self.shape = None
        self.stopped = False
        self._stop_signal = False
        self.occupied = False

        with self.queue.mutex:
            self.queue.queue.clear()

        self.start()

        while self.shape is None:
            time.sleep(0.01)

    def stop(self) -> None:
        """Stop this track's Stream"""
        self._stop_signal = True
        while not self.stopped:
            time.sleep(0.01)

    def start(self) -> None:
        """Start this track's Stream"""
        threading.Thread(target=self.__start, daemon=True).start()

    def _apply_fx(self, data) -> np.ndarray:
        # First apply the volume
        data = np.multiply(data, pow(
            2, (math.sqrt(math.sqrt(math.sqrt(self.vol))) * 192 - 192) / 6), casting="unsafe")
        return data

    def __callback(self, indata, outdata, frames, time, status) -> None:

        self.shape = outdata.shape

        try:
            data = self.queue.get(block=False)
            self.occupied = True
        except Empty:
            self.occupied = False
            data = None

        if self.occupied:
            outdata[:] = self._apply_fx(data)

    def __start(self) -> None:
        with sd.Stream(samplerate=self.samplerate, channels=2, callback=self.__callback):
            while not self._stop_signal:
                try:
                    time.sleep(0.001)
                except KeyboardInterrupt:
                    self.stop()

        self.stopped = True
        self._stop_signal = False


class Mixer:
    """
    Create a Mixer. This mixer will be responsible for handling all the tracks and playing audio.

    Parameters
    ----------
    `tracks` : List[Track]
        Your list of tracks.
    `conversion_path` : str
        Sometimes loading a file will fail due to it being an invalid format. To get around this, we convert it onto a .wav file, this is where those converted files are stored. Defaults to None which is to just allow loading of unsupported formats to fail.
    """

    def __init__(self, tracks: List[Track], conversion_path: str = None) -> None:
        self.tracks = tracks
        self.conversion_path = conversion_path

    def get_track(self, name: str = None, id_: int = None, require_unoccupied: bool = False) -> Union[None, Track]:
        """
        Retrieve a specific track either by its name or id.

        Parameters
        ----------
        `name` : str
            The name of the track.
        `id_` : int
            The id of the track (it's basically the index of the track in self.tracks)
        `require_unoccupied` : bool 
            Whether the track must be occupied. If this is True and we found a track but it's occupied, None will be return instead, otherwise the Track is returned. Defaults to False.

        Returns
        ------
        Either the Track if it was found or None.

        Raises
        ------
        `ValueError` :
            Raised when both name and id_ is not provided.
        """

        track = None

        if name:
            track = [x for x in self.tracks if x.name == name]
            if not track:
                return

            track = track[0]
        elif id_:
            try:
                track = self.tracks[id_]
            except IndexError:
                return
        else:
            raise ValueError(
                "one of the following parameters has to exist: name, id_")

        if require_unoccupied:
            if track.occupied:
                return
        return track

    def get_unoccupied_tracks(self) -> List[Track]:
        """Get a list of unoccupied tracks. Could be an empty list of no unoccupied tracks were found."""
        return [x for x in self.tracks if not x.occupied]

    def play_file(self, fp: str, **kwargs) -> None:
        """
        Play the provided file. The file will be split into chunks and is then put in the track's audio queue.

        Notes
        -----
        - If the provided file does not match the default sample rate, the track's samplerate will be modifed therefore causing the track to restart. This means if you already got something playing in a provided track, it will be stopped if the samplerate does not match the samplerate of the current track.
        - If the provided file's format is not supported. It will be converted into a .wav file and that .wav file is stored in self.conversion_path. If the conversion path is None then we will not try to convert it and just continue to raise an `UnsupportedFormat` error.

        Parameters
        ----------
        `fp`: str
            Path to the file.
        `track` : Track
            The track to use. If not provided, a unoccupied track will be used.

        Raises
        ------
        `NoUnoccupiedTrack` :
            Raised when there's no unoccupied track. Will not be raised if `track` is provided.
        `UnsupportedFormat` :
            Raised when the provided file format is not supported or when is it converted onto a .wav but it still fails.
        """

        # First get a available track
        track = kwargs.get("track")
        if track is None:
            track = self.get_unoccupied_tracks()
            if not track:
                raise NoUnoccupiedTrack
            track = track[0]

        try:
            _, rate = sf.read(fp, frames=1)
        except RuntimeError as e:
            if self.conversion_path is None:
                raise UnsupportedFormat(e)

            # Create if it doesn't exist
            Path(self.conversion_path).mkdir(parents=True, exist_ok=True)
            out = os.path.splitext(fp)[0] + ".wav"
            out = os.path.join(self.conversion_path, out)

            ff = ffmpy.FFmpeg(
                inputs={fp: None},
                outputs={out: None},
                global_options=["-loglevel", "quiet", "-y"]
            )

            try:
                ff.run()
                return self.play_file(out, **kwargs)
            except ffmpy.FFRuntimeError as e:
                raise UnsupportedFormat(e)

        track.update_samplerate(rate)
        blocksize = track.shape[0]

        nds = sf.blocks(
            fp,
            blocksize=blocksize,
            always_2d=True,
            fill_value=np.array([0]),
            dtype=np.float32
        )

        for nd in nds:
            track.queue.put(nd)
