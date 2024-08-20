#!/usr/bin/env python3

import numpy as np

def exr2array(img: str, chan: str) -> np.array:
    import Imath, OpenEXR
    handle = OpenEXR.InputFile(img)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_str = handle.channel(chan, pt)

    dw = handle.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channel_data = np.frombuffer(channel_str, dtype=np.float32)
    channel_data.shape = (height, width)
    return channel_data