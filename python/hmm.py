SKIP_CUM = True
PRINT_TRELLIS = False
import numpy as np

class TrellisItem:
    def __init__(self):
        self.max_prob = 0.0
        self.max_path = 0
        if not SKIP_CUM:
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
            if not SKIP_CUM:
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

                    if not SKIP_CUM:
                        hop_prob_cum = trellis[time - 1][state_a].cum_prob * self.prob_trans[state_a][state_b]
                        trellis[time][state_b].cum_prob += hop_prob_cum

                if not SKIP_CUM:
                    prob_emiss_local = self.prob_emiss_comp(state_b, observations[time])
                    trellis[time][state_b].cum_prob *= prob_emiss_local

            # normalize layer
            layer_total_max = sum([x.max_prob  for x in trellis[time]])
            for state_i in range(self.count_state):
                trellis[time][state_i].max_prob = trellis[time][state_i].max_prob / layer_total_max
            if not SKIP_CUM:
                layer_total_cum = sum([x.cum_prob for x in trellis[time]])
                for state_i in range(self.count_state):
                    trellis[time][state_i].cum_prob = trellis[time][state_i].cum_prob / layer_total_cum

        if PRINT_TRELLIS:
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
        
        if PRINT_TRELLIS:
            print("Most probable path probability: {max_path_prob:.4}")
            print("Most probable path: ")
            for time in range(len(observations)):
                print("{max_path[time]}-")
            print()

        if not SKIP_CUM:
            cum_path_prob = sum([x.cum_prob for x in trellis[observations.len() - 1]])
            if PRINT_TRELLIS:
                print("Cummulative observation probability: {cum_path_prob:.4}")

        return max_path