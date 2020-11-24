mod hmm;
mod example;


fn main() {
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