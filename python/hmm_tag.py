from hmm import HMM
import sys

class HMMTag(HMM):
    def __init__(self, loader):
        super().__init__(
            loader.mapper_t.count(),
            loader.mapper_w.count()
        )
        self.SCALE_FACTOR = 4096.0
        for sent in loader.data:
            key = sent.tokens[0][1]
            self.prob_start[key] += 1.0

        total = sum(self.prob_start) / self.SCALE_FACTOR
        for key in range(len(self.prob_start)):
            self.prob_start[key] /= total

        for sent in loader.data:
            for pos in range(1, len(sent.tokens)):
                key1 = sent.tokens[pos - 1][1]
                key2 = sent.tokens[pos][1]
                self.prob_trans[key1][key2] += 1.0

        for key1 in range(len(self.prob_trans)):
            total = sum(self.prob_trans[key1]) / self.SCALE_FACTOR
            for key2 in range(len(self.prob_trans[key1])):
                self.prob_trans[key1][key2] /= total

        for sent in loader.data:
            for pos in range(len(sent.tokens)):
                val = sent.tokens[pos][0]
                key = sent.tokens[pos][1]
                self.prob_emiss[key][val] += 1.0

        for key in range(len(self.prob_emiss)):
            total = sum(self.prob_emiss[key]) / self.SCALE_FACTOR
            for val in range(len(self.prob_emiss[key])):
                self.prob_emiss[key][val] /= total

    def eval_tag(self, loader):
        total = 0
        correct = 0
        for sent in loader.data:
            observations = [x[0] for x in sent.tokens]
            total += len(observations)
            max_path = self.viterbi(observations)

            for time in range(len(observations)):
                if sent.tokens[time][1] == max_path[time]:
                    correct += 1

            if 'print_pred' in sys.argv:
                for time in range(len(max_path)):
                    print(
                        f"{loader.mapper_w.map_from[sent.tokens[time][0]]}\t" +
                        f"{loader.mapper_t.map_from[max_path[time]]}"
                    )
                print()

        if "print_acc" in sys.argv:
            acc = correct/total
            print(f"- Accuracy: {acc * 100:.2f}%")
