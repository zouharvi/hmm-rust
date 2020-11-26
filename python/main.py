from loader import Loader
from hmm import HMM
from hmm_tag import HMMTag

if __name__ == '__main__':
    print("Loading data")
    data_train = Loader(None, "data/de-train.tt")
    data_eval = Loader(data_train, "data/de-eval.tt")
    print("Fitting the model")
    model = HMMTag(data_train)
    # print("Train dataset:")
    # model.eval_tag(data_train)
    print("Dev dataset:")
    model.eval_tag(data_eval)