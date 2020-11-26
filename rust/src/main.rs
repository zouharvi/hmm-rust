mod hmm;
mod hmm_ice;
mod hmm_tag;
mod loader;
use loader::Loader;

fn main() {
    println!("Loading data");
    let data_train = Loader::load("../data/de-train.tt").unwrap();
    let data_eval = Loader::load_from_loader(&data_train, "../data/de-eval.tt").unwrap();
    println!("Fitting the model");
    let mut model = hmm::HMM::hmm_tag(&data_train);
    println!("Train dataset:");
    model.eval_tag(&data_train);
    println!("Dev dataset:");
    model.eval_tag(&data_eval);
}

#[test]
fn ice_example() {
    let mut model = hmm::HMM::hmm_ice();
    println!("-----");
    model.traverse(3);
    println!("-----");
    model.viterbi(vec![1, 2, 0]);
}
