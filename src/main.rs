mod hmm;
mod hmm_ice;
mod hmm_tag;
mod loader;
use loader::Loader;

fn main() {
    #[cfg(feature="skip_cum")] {
        println!("Skipping cummulative probability computation")
    }

    println!("Loading data");
    let data_train = Loader::load("data/de-train.tt").unwrap();
    // let data_eval = Loader::load("data/de-eval.tt").unwrap();
    println!("Fitting the model");
    let mut model = hmm::HMM::hmm_tag(&data_train);
    println!("Evaluation");
    model.eval_tag(&data_train);
}

#[test]
fn ice_example() {
    let mut model = hmm::HMM::hmm_ice();
    println!("-----");
    model.traverse(3);
    println!("-----");
    model.viterbi(vec![1, 2, 0]);
}
