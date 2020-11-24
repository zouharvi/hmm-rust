mod hmm;
mod hmm_tag;
mod hmm_ice;
mod loader;
use loader::Loader;

fn main() {
    // load data
    println!("Loading data");
    let data_train = Loader::load("data/de-eval.tt").unwrap();
    // data_train.print();
    println!("Fitting the model");
    let model = hmm::HMM::hmm_tag(&data_train);
}

#[test]
fn ice_example() {
    let mut model = hmm::HMM::hmm_ice();
    println!("-----");
    model.traverse(3);
    println!("-----");
    model.viterbi(vec![1, 2, 0], true);
}