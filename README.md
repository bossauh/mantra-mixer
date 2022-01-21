# Mantra Mixer
Audio mixing library simplified. This library was made specifically for [Mantra AI](https://github.com/bossauh/mantra-ai) and still lacks a lot of features.

**What can you do with this?**

Well, you can play different audio files at the exact same time. It achieves this by creating tracks and each audio is being played at a separate thread.

## Requirements
- FFMpeg

## Usage
```py
import asyncio

from mantra_mixer import Mixer, Track


if __name__ == "__main__":
    async def main() -> None:
        # First let's create 4 tracks because we want to play 4 tracks at the same time
        # This creates 4 tracks with the name being the index but a string
        tracks = [Track(str(i)) for i in range(4)]

        # Now let's provide these tracks to the mixer
        mixer = Mixer(tracks)

        # Alternatively, the mixer can do this automatically for you by providing a int instead that tells the mixer how many tracks to generate.
        mixer = Mixer(4)

        # Optionally, we can provide a conversion_path folder to the Mixer class.
        # Sometimes a file format won't be supported, so we can get around this by converting it to a .wav file using ffpmeg.
        # The converted file is then put into the conversion_path folder.
        mixer = Mixer(tracks, conversion_path="./converted_files")

        # The following files will all be played at the same time
        await mixer.play_file("./file_1.mp3")
        await mixer.play_file("./file_2.mp3")
        await mixer.play_file("./different_format_file_3.m4a")
        await mixer.play_file("./different_format_file_4.flac")

        # These functions are non blocking functions. If you want them to block, pass a blocking=True
        await mixer.play_file("./blocking.mp3", blocking=True)
        print("This line won't be reached until blocking.mp3 is finished playing")

        # If you want the files to be loaded in memory you can do so by passing load_in_memory=True
        # You usually won't do this to avoid large memory usages, but sometimes you may want this because let's say after calling play_file, you wat to delete the file immediately or replace it.
        await mixer.play_file("./loaded_in_memory.mp3", load_in_memory=True)

    asyncio.run(main())
```

# LICENSE
MIT License
Copyright (c) 2022 Philippe Mathew
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
