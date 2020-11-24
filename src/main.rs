mod hmm;
mod example;
mod loader;
use loader::{Loader,Mapper};

fn main() {
    // load data
    let data_train = Loader::load("data/de-train.tt").unwrap();
    data_train.print();
}


#[test]
fn ice_example() {
    let mut ice = hmm::HMM {
        data: hmm::HMMData::example_ice(),
        state: 0,
    };
    println!("-----");
    ice.traverse(3);
    println!("-----");
    ice.viterbi(vec![1, 2, 0]);
}