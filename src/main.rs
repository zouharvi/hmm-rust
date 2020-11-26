mod hmm;
mod hmm_ice;
mod hmm_tag;
mod loader;
use loader::Loader;

fn main() {
    #[cfg(feature = "skip_cum")] {
        println!("Skipping cummulative probability computation")
    }

    println!("Loading data");
    let data_train = Loader::load("data/de-train.tt").unwrap();
    let data_eval = Loader::load_from_loader(&data_train, "data/de-eval.tt").unwrap();
    println!("Fitting the model");
    let mut model = hmm::HMM::hmm_tag(&data_train);
    println!("Evaluation");
    let acc_train = model.eval_tag(&data_train);
    let acc_eval = model.eval_tag(&data_eval);

    #[cfg(feature = "acc_print")] {
        println!("Train accuracy: {:.2}%", acc_train*100.0);
        println!("Eval  accuracy: {:.2}%", acc_eval*100.0);
    }

}

#[test]
fn ice_example() {
    let mut model = hmm::HMM::hmm_ice();
    println!("-----");
    model.traverse(3);
    println!("-----");
    model.viterbi(vec![1, 2, 0]);
}
