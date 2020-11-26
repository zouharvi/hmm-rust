from loader import Loader
from hmm import HMM
from hmm_tag import HMMTag
import sys

if __name__ == '__main__':
    sys.argv = [x.lower() for x in sys.argv]

    print("Loading data", file=sys.stderr)
    data_train = Loader(None, "data/de-train.tt")
    print("Fitting the model", file=sys.stderr)
    model = HMMTag(data_train)
    
    if 'comp_train' in sys.argv:
        print("Train dataset:", file=sys.stderr)
        model.eval_tag(data_train, True)
    if 'comp_dev' in sys.argv:
        print("Dev dataset:", file=sys.stderr)
        data_eval = Loader(data_train, "data/de-eval.tt")
        model.eval_tag(data_eval, True)
    if 'comp_test' in sys.argv:
        print("Test dataset:", file=sys.stderr)
        data_test = Loader(data_train, "data/de-test.t")
        model.eval_tag(data_test, False)
