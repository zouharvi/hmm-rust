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
Trellis unit (cum_prob, max_prob, max_pointer)
(0.20, 0.20, 0) (0.10, 0.05, 1) (0.01, 0.00, 0) 
(0.25, 0.25, 0) (0.00, 0.00, 1) (0.02, 0.01, 0) 
Cummulative observation probability: 0.0235
Most probable path probability: 0.0141
Most probable path: 1-0-1-
```

Absolutely don't use this code anywhere.