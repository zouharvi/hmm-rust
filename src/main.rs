mod hmm;
use hmm::{HMMData, HMM};

fn main() {
    let mut ice = HMM {
        data: HMMData::example_ice(),
        state: 0,
    };
    println!("-----");
    ice.traverse(3);
    println!("-----");
    ice.viterbi(vec![1, 2, 0]);
}
