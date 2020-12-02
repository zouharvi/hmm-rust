use crate::hmm::HMM;
use crate::loader::Loader;

#[allow(dead_code)]
impl HMM {
    // estimate HMM parameters given loaded corpus
    pub fn hmm_tag(loader: &Loader) -> HMM {
        let mut hmm = HMM::zeroes(
            loader.mapper_t.count().unwrap(),
            loader.mapper_w.count().unwrap(),
        );
        for sent in &loader.data {
            let key = sent.tokens[0].1;
            hmm.prob_start[key] += 1.0;
        }
        let total: f64 = hmm.prob_start.iter().sum::<f64>();
        for key in 0..hmm.prob_start.len() {
            hmm.prob_start[key] /= total;
        }
        for sent in &loader.data {
            for pos in 1..sent.tokens.len() {
                let key1 = sent.tokens[pos - 1].1;
                let key2 = sent.tokens[pos].1;
                hmm.prob_trans[key1][key2] += 1.0;
            }
        }

        for key1 in 0..hmm.prob_trans.len() {
            let total: f64 = hmm.prob_trans[key1].iter().sum::<f64>();
            for key2 in 0..hmm.prob_trans[key1].len() {
                hmm.prob_trans[key1][key2] /= total;
            }
        }

        #[cfg(feature = "smooth")]
        {
            for key1 in 0..hmm.prob_trans.len() {
                for key2 in 0..hmm.prob_trans[key1].len() {
                    hmm.prob_trans[key1][key2] += 0.001;
                }
            }
        }

        for sent in &loader.data {
            for pos in 0..sent.tokens.len() {
                let val = sent.tokens[pos].0;
                let key = sent.tokens[pos].1;
                hmm.prob_emiss[key][val] += 1.0;
            }
        }
        for key in 0..hmm.prob_emiss.len() {
            let total: f64 = hmm.prob_emiss[key].iter().sum::<f64>();
            for val in 0..hmm.prob_emiss[key].len() {
                hmm.prob_emiss[key][val] /= total;
            }
        }

        hmm
    }
    // evaluate or print predictions
    pub fn eval_tag(&mut self, loader: &Loader) {
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

            #[cfg(feature = "print_pred")]
            {
                for time in 0..max_path.len() {
                    println!(
                        "{}\t{}",
                        loader.mapper_w.map_from.get(&sent.tokens[time].0).unwrap(),
                        loader.mapper_t.map_from.get(&max_path[time]).unwrap(),
                    );
                }
                println!();
            }
        }

        #[cfg(feature = "print_acc")]
        {
            let acc = (correct as f64) / (total as f64);
            println!("- Accuracy: {:.2}%", acc * 100.0);
        }
    }
}
