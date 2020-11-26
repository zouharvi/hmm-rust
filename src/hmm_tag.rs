use crate::hmm::HMM;
use crate::loader::Loader;

impl HMM {
    pub fn hmm_tag(loader: &Loader) -> HMM {
        let mut hmm = HMM::zeroes(
            loader.mapper_t.count().unwrap() + 1,
            loader.mapper_w.count().unwrap() + 1,
        );

        println!("Start probabilities");
        for sent in &loader.data {
            let key = sent.tokens[0].1;
            hmm.prob_start[key] += 1.0;
        }
        let total: f64 = hmm.prob_start.iter().sum::<f64>() / 4096.0;
        for key in 0..hmm.prob_start.len() {
            hmm.prob_start[key] /= total;
        }

        println!("Transition probabilities");
        for sent in &loader.data {
            for pos in 1..sent.tokens.len() {
                let key1 = sent.tokens[pos - 1].1;
                let key2 = sent.tokens[pos].1;
                hmm.prob_trans[key1][key2] += 1.0;
            }
        }
        for key1 in 0..hmm.prob_trans.len() {
            let total: f64 = hmm.prob_trans[key1].iter().sum::<f64>() / 4096.0;
            for key2 in 0..hmm.prob_trans[key1].len() {
                hmm.prob_trans[key1][key2] /= total;
            }
        }

        println!("Emission probabilities");
        for sent in &loader.data {
            for pos in 0..sent.tokens.len() {
                let val = sent.tokens[pos].0;
                let key = sent.tokens[pos].1;
                hmm.prob_emiss[key][val] += 1.0;
            }
        }
        for key in 0..hmm.prob_emiss.len() {
            let total: f64 = hmm.prob_emiss[key].iter().sum::<f64>() / 4096.0;
            for val in 0..hmm.prob_emiss[key].len() {
                hmm.prob_emiss[key][val] /= total;
            }
        }

        return hmm;
    }

    pub fn eval_tag(&mut self, loader: &Loader) -> f64 {
        let mut total = 0;
        let mut correct = 0;
        for sent in &loader.data {
            let observations = sent.tokens.iter().map(|x| x.0).collect::<Vec<usize>>();
            total += observations.len();
            let max_path = self.viterbi(&observations);

            for time in 0..observations.len() {
                if sent.tokens[time].1 == max_path[time] {
                    correct += 1;
                }
            }

            #[cfg(feature = "sent_pred_print")]
            {
                print!("Pred: ");
                for time in 0..max_path.len() {
                    print!("{}-", max_path[time]);
                }
                println!();
                print!("True: ");
                for time in 0..max_path.len() {
                    print!("{}-", sent.tokens[time].1);
                }
                println!();
            }
        }

        return (correct as f64) / (total as f64)
    }
}
