use rand::rngs::ThreadRng;
use rand::Rng;

pub struct HMM {
    count_state: usize,
    count_emiss: usize,
    pub prob_start: Vec<f32>,
    pub prob_trans: Vec<Vec<f32>>,
    pub prob_emiss: Vec<Vec<f32>>,
    state: usize,
}

impl HMM {
    pub fn zeroes(count_state: usize, count_emiss: usize) -> HMM {
        let prob_start = vec![0.0; count_state];
        let prob_trans = vec![vec![0.0; count_state]; count_state];
        let prob_emiss = vec![vec![0.0; count_emiss]; count_state];

        HMM {
            count_state,
            count_emiss,
            prob_start,
            prob_trans,
            prob_emiss,
            state: 0,
        }
    }
}

#[derive(Clone)]
struct TrellisItem {
    #[cfg(not(feature = "skip_cum"))]
    pub cum_prob: f32,
    pub max_prob: f32,
    pub max_path: usize,
}

impl TrellisItem {
    pub fn new() -> TrellisItem {
        TrellisItem {
            #[cfg(not(feature = "skip_cum"))]
            cum_prob: 0.0,
            max_prob: 0.0,
            max_path: 0,
        }
    }
}

impl HMM {
    #[allow(dead_code)]
    fn vector_sample(probs: &Vec<f32>, mut rng: ThreadRng) -> Option<usize> {
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

    #[allow(dead_code)]
    pub fn traverse(&mut self, steps: usize) {
        let rng = rand::thread_rng();
        self.state = HMM::vector_sample(&self.prob_start, rng).unwrap();
        for _ in 0..steps {
            let emission = HMM::vector_sample(&self.prob_emiss[self.state], rng).unwrap();
            println!("State: {}, Emission: {}", self.state, emission);

            self.state = HMM::vector_sample(&self.prob_trans[self.state], rng).unwrap();
        }
    }

    fn prob_emiss_comp(&self, state: usize, observation: usize) -> f32 {
        if observation >= self.prob_emiss[state].len() {
            return 1.0;
        } else {
            return self.prob_emiss[state][observation];
        }
    }

    pub fn viterbi(&mut self, observations: Vec<usize>) -> Vec<usize> {
        let mut trellis = vec![vec![TrellisItem::new(); self.count_state]; observations.len()];
        for state in 0..(self.count_state) {
            let hop_prob = self.prob_start[state] * self.prob_emiss_comp(state, observations[0]);
            trellis[0][state].max_prob = hop_prob;
            #[cfg(not(feature = "skip_cum"))]
            {
                trellis[0][state].cum_prob = hop_prob;
            }
        }

        // Trellis computation
        for time in 1..observations.len() {
            for state_b in 0..(self.count_state) {
                for state_a in 0..(self.count_state) {
                    let hop_prob =
                        trellis[time - 1][state_a].max_prob * self.prob_trans[state_a][state_b];
                    if hop_prob > trellis[time][state_b].max_prob {
                        trellis[time][state_b].max_path = state_a;
                        trellis[time][state_b].max_prob = hop_prob;
                    }
                    let prob_emiss_local = self.prob_emiss_comp(state_b, observations[time]);
                    trellis[time][state_b].max_prob *= prob_emiss_local;
                }
                #[cfg(not(feature = "skip_cum"))]
                {
                    for state_a in 0..(self.count_state) {
                        let hop_prob =
                            trellis[time - 1][state_a].cum_prob * self.prob_trans[state_a][state_b];
                        let prob_emiss_local = self.prob_emiss_comp(state_b, observations[time]);
                        trellis[time][state_b].cum_prob += hop_prob * prob_emiss_local;
                    }
                }
            }
        }

        #[cfg(feature = "trellis_print")]
        {
            // Trellis printing
            println!("Trellis unit (max_prob, max_pointer)");
            for state in 0..self.count_state {
                for time in 0..observations.len() {
                    print!(
                        "({:.2}, {}) ",
                        trellis[time][state].max_prob,
                        trellis[time][state].max_path,
                    );
                }
                println!();
            }
        }

        let (max_path_end, max_path_prob) = trellis[observations.len() - 1]
            .iter()
            .enumerate()
            .max_by(|(_, value0), (_, value1)| {
                value0.max_prob.partial_cmp(&value1.max_prob).unwrap()
            })
            .map(|(idx, val)| (idx, val.max_prob))
            .unwrap();
        let mut max_path = vec![0; observations.len()];
        max_path[observations.len() - 1] = max_path_end;
        for time in (0..(observations.len() - 1)).rev() {
            max_path[time] = trellis[time + 1][max_path[time + 1]].max_path;
        }
        #[cfg(feature = "trellis_print")]
        {
            println!("Most probable path probability: {:.4}", max_path_prob);
            print!("Most probable path: ");
            for time in 0..observations.len() {
                print!("{}-", max_path[time]);
            }
            println!();
        }

        #[cfg(not(feature = "skip_cum"))]
        {
            let cum_path_prob = trellis[observations.len() - 1]
                .iter()
                .max_by(|value0, value1| value0.cum_prob.partial_cmp(&value1.cum_prob).unwrap())
                .map(|val| val.cum_prob)
                .unwrap();
            #[cfg(feature = "trellis_print")]
            {
                println!("Cummulative observation probability: {:.4}", cum_path_prob);
            }
        }

        return max_path;
    }
}
