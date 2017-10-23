#!/usr/bin/env python3

from glob import glob
from io import BytesIO
from PIL import Image
import struct


for fn in glob('data/orig/*'):
    bmp = BytesIO()
    img = Image.open(open(fn,'rb'))
    img.save(bmp, 'BMP')
    bmp = bmp.getvalue()
    w = struct.unpack("<L", bmp[18:22])[0]
    h = struct.unpack("<L", bmp[22:26])[0]
    ratio = 1.5
    l,t = 0,0
    dw = int(h*ratio)
    dh = int(w/ratio)
    if w > dw:
        l = (w-dw)//2
        w = dw
    elif h > dh:
        t = (h-dh)//2
        h = dh
    ofn = fn.replace('/orig', '')
    print(ofn)
    img.crop((l, t, w, h)).resize((60,40),Image.ANTIALIAS).convert('L').save(ofn)
