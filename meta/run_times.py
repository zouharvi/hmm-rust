#!/usr/bin/env python3

import subprocess
import os
import time

def safe_rm(path):
    if os.path.exists(path): os.remove(path)

safe_rm('data/de-train-new.tt')
safe_rm('data_measured/time')

_, _ = subprocess.Popen(
    "make r-build-time".split(),
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL
).communicate()

with open('data/de-train.tt', 'r') as f:
    lines = f.readlines()

for i in range(1,len(lines)//10000+2):
    print(i)
    with open('data/de-train-new.tt', 'w') as f:
        f.writelines(lines[:i*10000])
        start = time.time()
        process = subprocess.Popen("make r-run".split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        output, _ = process.communicate()
        end = time.time()

        output = output.decode('utf-8').replace('- Accuracy: ', '').replace('\n', ',').replace('%', '')
        output1 = f'{end-start},{output}'

        start = time.time()
        process = subprocess.Popen("make p-run-time".split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        output, _ = process.communicate()
        end = time.time()
        output = output.decode('utf-8').replace('- Accuracy: ', '').replace('\n', ',').replace('%', '')
        output2 = f'{end-start},{output}'

        with open('data_measured/time', 'a') as f:
            f.write(str(i*10000) + ',' + output1+output2+'\n')

safe_rm('tmp-acc')
safe_rm('tmp-time')
safe_rm('data/de-train-new.tt')
