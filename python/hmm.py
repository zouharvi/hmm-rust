import numpy as np
import sys

class TrellisItem:
    def __init__(self):
        self.max_prob = 0.0
        self.max_path = 0
        if "comp_cum" in sys.argv:
            self.cum_prob = 0.0

class HMM:
    def __init__(self, count_state, count_emiss):
        self.prob_start = [0.0]*count_state
        self.prob_trans = [[0.0]*count_state for _ in range(count_state)]
        self.prob_emiss = [[0.0]*count_emiss for _ in range(count_state)]
        self.state = 0
        self.count_state = count_state
        self.count_emiss = count_emiss

    def prob_emiss_comp(self, state, observation):
        if observation >= len(self.prob_emiss[state]):
            return 1.0
        else:
            return self.prob_emiss[state][observation]

    def viterbi(self, observations):
        trellis = [ [TrellisItem() for _ in range(self.count_state)] for _ in observations]
        
        # Initial hop
        for state in range(self.count_state):
            hop_prob = self.prob_start[state] * self.prob_emiss_comp(state, observations[0])
            trellis[0][state].max_prob = hop_prob
            if "comp_cum" in sys.argv:
                trellis[0][state].cum_prob = hop_prob

        # Trellis computation
        for time in range(1, len(observations)):
            for state_b in range(self.count_state):
                for state_a in range(self.count_state):
                    hop_prob_max = trellis[time - 1][state_a].max_prob * self.prob_trans[state_a][state_b]
                    if hop_prob_max > trellis[time][state_b].max_prob:
                        trellis[time][state_b].max_path = state_a
                        trellis[time][state_b].max_prob = hop_prob_max

                    prob_emiss_local = self.prob_emiss_comp(state_b, observations[time])
                    trellis[time][state_b].max_prob *= prob_emiss_local

                    if "comp_cum" in sys.argv:
                        hop_prob_cum = trellis[time - 1][state_a].cum_prob * self.prob_trans[state_a][state_b]
                        trellis[time][state_b].cum_prob += hop_prob_cum

                if "comp_cum" in sys.argv:
                    prob_emiss_local = self.prob_emiss_comp(state_b, observations[time])
                    trellis[time][state_b].cum_prob *= prob_emiss_local

            # normalize layer
            layer_total_max = sum([x.max_prob  for x in trellis[time]])
            for state_i in range(self.count_state):
                if layer_total_max != 0:
                    trellis[time][state_i].max_prob = trellis[time][state_i].max_prob / layer_total_max
                else:
                    trellis[time][state_i].max_prob = 1
            if "comp_cum" in sys.argv:
                layer_total_cum = sum([x.cum_prob for x in trellis[time]])
                for state_i in range(self.count_state):
                    trellis[time][state_i].cum_prob = trellis[time][state_i].cum_prob / layer_total_cum

        if "print_trellis" in sys.argv:
            print("Trellis unit (max_prob, max_pointer)")
            for state in range(self.count_state):
                for time in range(len(observations)):
                    print(f"{trellis[time][state].max_prob:.2f}, {trellis[time][state].max_path}")
                print()

        max_path_end = np.argmax([x.max_prob for x in trellis[len(observations)-1]])
        max_path_prob = trellis[len(observations)-1][max_path_end]
        max_path = [0]*len(observations)
        max_path[len(observations) - 1] = max_path_end

        for time in range(len(observations)-1):
            max_path[time] = trellis[time + 1][max_path[time + 1]].max_path
        
        if "print_trellis" in sys.argv:
            print("Most probable path probability: {max_path_prob:.4}")
            print("Most probable path: ")
            for time in range(len(observations)):
                print("{max_path[time]}-")
            print()

        if "comp_cum" in sys.argv:
            cum_path_prob = sum([x.cum_prob for x in trellis[len(observations) - 1]])
            if "print_trellis" in sys.argv:
                print("Cummulative observation probability: {cum_path_prob:.4}")

        return max_path