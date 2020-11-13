use rand::Rng;

pub struct HMMData {
    count_state: usize,
    count_emiss: usize,
    prob_start: Vec<f32>,
    prob_trans: Vec<Vec<f32>>,
    prob_emiss: Vec<Vec<f32>>,
}

impl HMMData {
    fn zeroes(count_state: usize, count_emiss: usize) -> HMMData {
        let mut prob_start = Vec::<f32>::new();
        for _ in 0..count_state {
            prob_start.push(0.0);
        }
        let mut prob_trans = Vec::<Vec<f32>>::new();
        for _ in 0..count_state {
            prob_trans.push(vec![0.0; count_state]);
        }
        let mut prob_emiss = Vec::<Vec<f32>>::new();
        for _ in 0..count_emiss {
            prob_emiss.push(vec![0.0; count_emiss]);
        }

        HMMData {
            count_state,
            count_emiss,
            prob_start,
            prob_trans,
            prob_emiss,
        }
    }

    pub fn example_ice() -> HMMData {
        let mut hmm = HMMData::zeroes(2, 3);

        // 0 - hot, 1 - cold
        hmm.prob_start[0] = 0.5;
        hmm.prob_start[1] = 0.5;
        hmm.prob_trans[0] = vec![0.6, 0.4];
        hmm.prob_trans[1] = vec![0.5, 0.5];
        // 0 - 1 ice cream, 1 - 2 ice creams, 2 - 3 ice creams
        hmm.prob_emiss[0] = vec![0.2, 0.4, 0.4];
        hmm.prob_emiss[1] = vec![0.6, 0.5, 0.0];

        return hmm;
    }
}

pub struct HMM {
    pub data: HMMData,
    pub state: usize,
}

impl HMM {
    fn roll_vector(probs: &Vec<f32>, mut rng: rand::rngs::ThreadRng) -> Option<usize> {
        let dice = rng.gen::<f32>();
        let mut cum = 0.0;
        for n in 0..(probs.len()) {
            cum += probs[n];
            if dice <= cum {
                return Some(n);
            }
        }
        return None;
    }

    pub fn traverse(&mut self, steps: usize) {
        let rng = rand::thread_rng();
        self.state = HMM::roll_vector(&self.data.prob_start, rng).unwrap();
        for _ in 0..steps {
            let emission = HMM::roll_vector(&self.data.prob_emiss[self.state], rng).unwrap();
            println!("State: {}, Emission: {}", self.state, emission);

            self.state = HMM::roll_vector(&self.data.prob_trans[self.state], rng).unwrap();
        }
    }

    pub fn viterbi(&mut self, observations: Vec<usize>) {
        let mut paths = Vec::<Vec<(usize, f32)>>::new();
        for _ in 0..observations.len() {
            paths.push(vec![(0, 0.0); self.data.count_state]);
        }

        for state in 0..(self.data.count_state) {
            let hop_prob =
                self.data.prob_start[state] * self.data.prob_emiss[state][observations[0]];
            paths[0][state] = (0, hop_prob);
        }

        for time in 1..observations.len() {
            for state_b in 0..(self.data.count_state) {
                let mut max_hop = 0.0;
                for state_a in 0..(self.data.count_state) {
                    let hop_prob =
                        paths[time - 1][state_a].1 * self.data.prob_trans[state_a][state_b];
                    if hop_prob > max_hop {
                        paths[time][state_b].0 = state_a;
                        max_hop = hop_prob;
                    }
                    paths[time][state_b].1 += hop_prob * self.data.prob_emiss[state_b][observations[time]];
                }
            }
        }
        for state in 0..self.data.count_state {
            for time in 0..observations.len() {
                print!("{} ({:.2}), ", paths[time][state].0, paths[time][state].1);
            }
            println!();
        }

        let mut max_tup = (0, 0.0);
        for path_tup in &paths[observations.len()-1] {
            if path_tup.1 > max_tup.1 {
                max_tup.0 = path_tup.0;
                max_tup.1 = path_tup.1;
            }
        }
        
        println!("Most probable path with probability {:.2}:", max_tup.1);
        let mut max_path = vec![0; observations.len()];
        max_path[observations.len()-1] = max_tup.0; 
        for time in (0..(observations.len()-1)).rev() {
            max_path[time] = paths[time+1][max_path[time+1]].0;
        }
        for time in 0..observations.len() {
            print!("{}-", max_path[time]);
        }
        println!();
    }
}
