#!/usr/bin/env python3

import subprocess
import os
import time

import matplotlib.pyplot as plt
# time-3
fig, ax1 = plt.subplots(figsize=(6, 4))
plt.title('Train + Compute Eval. + Compute Train (3)')

with open('data_measured/time-3', 'r') as f:
    data = [x.rstrip('\n').split(',') for x in f.readlines()]
v_size = [float(x[0]) for x in data]
v_run_r = [float(x[1]) for x in data]
v_run_rta = [float(x[2])/100 for x in data]
v_run_rea = [float(x[3])/100 for x in data]
v_run_p = [float(x[4]) for x in data]
v_run_pta = [float(x[5])/100 for x in data]
v_run_pea = [float(0 if x[6] == '' else x[6])/100 for x in data]
a_p, = ax1.plot(v_size, v_run_p, color='green', linestyle='--')
a_r, = ax1.plot(v_size, v_run_r, color='orange', linestyle='--')

ax2 = ax1.twinx()

a_pea, = ax2.plot(v_size, v_run_pea, color='green', linestyle=':')
a_rea, = ax2.plot(v_size, v_run_rea, color='orange', linestyle=':')
a_pta, = ax2.plot(v_size, v_run_pta, color='green', linestyle='-.')
a_rta, = ax2.plot(v_size, v_run_rta, color='orange', linestyle='-.')

plt.legend(
    [a_pea, a_rea, a_pta, a_rta, a_p, a_r],
    [
        'Python eval. accuracy', 'Rust eval. accuracy', 'Python train accuracy',
        'Rust train accuracy', 'Python time', 'Rust time'
    ]
)
ax1.set_ylabel('Time (s)')
ax2.set_ylabel('Accuracy')
ax1.set_xlabel('Train token count')
plt.show()


# time-2
fig, ax1 = plt.subplots(figsize=(6, 4))
plt.title('Train + Compute Eval. (2)')

with open('data_measured/time-2', 'r') as f:
    data = [x.rstrip('\n').split(',') for x in f.readlines()]
v_size = [float(x[0]) for x in data]
v_run_r = [float(x[1]) for x in data]
v_run_rea = [float(x[2])/100 for x in data]
v_run_p = [float(x[3]) for x in data]
v_run_pea = [float(x[4])/100 for x in data]
a_p, = ax1.plot(v_size, v_run_p, color='green', linestyle='--')
a_r, = ax1.plot(v_size, v_run_r, color='orange', linestyle='--')

ax2 = ax1.twinx()

a_pea, = ax2.plot(v_size, v_run_pea, color='green', linestyle=':')
a_rea, = ax2.plot(v_size, v_run_rea, color='orange', linestyle=':')

plt.legend(
    [a_pea, a_rea, a_p, a_r],
    ['Python eval. accuracy', 'Rust eval. accuracy', 'Python time', 'Rust time']
)
ax1.set_ylabel('Time (s)')
ax2.set_ylabel('Accuracy')
ax1.set_xlabel('Train token count')
plt.show()


# time-1
with open('data_measured/time-1', 'r') as f:
    data = [x.rstrip('\n').split(',') for x in f.readlines()]
v_size = [float(x[0]) for x in data]
v_run_r = [float(x[1]) for x in data]
v_run_p = [float(x[2]) for x in data]

fig, ax1 = plt.subplots(figsize=(6, 4))
plt.title('Train only (1)')

plt.plot(v_size, v_run_p, label='Python time', color='green')
plt.plot(v_size, v_run_r, label='Rust time', color='orange')

plt.legend()
plt.ylabel('Time (s)')
plt.xlabel('Train token count')
plt.show()
