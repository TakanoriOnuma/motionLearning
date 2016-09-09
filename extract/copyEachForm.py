# -*- conding: utf-8 -*-
import os
import shutil

IMG_DIR = 'extract'
ROOT    = 'images5/normal/test'
fForm = open('form.txt', 'r')
i = 0
for line in fForm:
    start, end = line[:-1].split('\t')
    print start, end
    if not os.path.exists('{}/{}'.format(ROOT, i)):
        os.mkdir('{}/{}'.format(ROOT, i))
    for idx in range(int(start), int(end) + 1):
        src  = '{}/img{}.png'.format(IMG_DIR, idx)
        dest = '{}/{}/img{}.png'.format(ROOT, i, idx)
        shutil.copyfile(src, dest)

    i += 1
