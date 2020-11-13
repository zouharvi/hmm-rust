# Rust HMM

Just a few hours toy implementation of random HMM walk and the viterbi algorithm.

Example output on the famous ice cream consumption problem on hot and cold days:

```
State: 1, Emission: 1
State: 0, Emission: 2
State: 1, Emission: 0
```

With the observations `1 2 0` the viterbi algorithm outputs this:

```
0 (0.20), 1 (0.10), 0 (0.01) 
0 (0.25), 1 (0.00), 0 (0.02) 
Most probable path with probability 0.02:
1-0-0
```

Absolutely don't use this code anywhere.