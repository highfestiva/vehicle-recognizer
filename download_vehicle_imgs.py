#!/usr/bin/env python3

from glob import glob
import requests


def search_n(q,s):
    for extra in ['', '&u=yahoo', '&u=yahoo&f=,,,']:
        try:
            url = 'https://duckduckgo.com/i.js?q=%s&o=json&p=1&s=%i%s&l=wt-wt'%(q,s,extra)
            r = requests.get(url)
            return r.json()['results']
        except:
            pass
    print(url)
    print(r)
    print(r.text)
    return []


def imgs_links(q):
    rs = [r for s in range(50,300,50) for r in search_n(q,s)]
    return [hit['thumbnail'] for hit in rs]


def download_imgs(q):
    mime2ext = {'image/jpeg': 'jpg'}
    for i,imgurl in enumerate(imgs_links(q)):
        ext = '*'
        fn = 'data/orig/%s%3.3i.%s' % (q,i,ext)
        if glob(fn):
            continue
        r = requests.get(imgurl)
        ext = mime2ext[r.headers['Content-Type']]
        fn = 'data/orig/%s%3.3i.%s' % (q,i,ext)
        print(fn)
        with open(fn,'wb') as f:
            f.write(r.content)


download_imgs('car')
download_imgs('motorcycle')
download_imgs('bicycle')
