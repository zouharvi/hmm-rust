mod hmm;
mod hmm_ice;
mod hmm_tag;
mod loader;
use loader::Loader;

fn main() {
    eprintln!("Loading data");
    let data_train: Loader;
    if cfg!(feature = "new_train") {
        data_train = Loader::load("data/de-train-new.tt").unwrap();
    } else {
        data_train = Loader::load("data/de-train.tt").unwrap();
    }
    eprintln!("Fitting the model");
    let mut model = hmm::HMM::hmm_tag(&data_train);

    #[cfg(feature = "comp_train")]
    {
        eprintln!("Train dataset:");
        model.eval_tag(&data_train);
    }

    #[cfg(feature = "comp_eval")]
    {
        eprintln!("Eval dataset:");
        let data_eval = Loader::load_from_loader(&data_train, "data/de-eval.tt").unwrap();
        model.eval_tag(&data_eval);
    }

    #[cfg(feature = "comp_test")]
    {
        eprintln!("Test dataset:");
        let data_test = Loader::load_from_loader(&data_train, "data/de-test.t").unwrap();
        model.eval_tag(&data_test);
    }
}

#[test]
fn ice_example() {
    let mut model = hmm::HMM::hmm_ice();
    println!("-----");
    model.traverse(3);
    println!("-----");
    model.viterbi(&vec![1, 2, 0]);
}
