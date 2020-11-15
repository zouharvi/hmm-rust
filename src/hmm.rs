use rand::rngs::ThreadRng;
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

#[derive(Clone)]
struct TrellisItem {
    pub cum_prob: f32,
    pub max_prob: f32,
    pub max_path: usize,
}

impl TrellisItem {
    pub fn new() -> TrellisItem {
        TrellisItem {
            cum_prob: 0.0,
            max_prob: 0.0,
            max_path: 0,
        }
    }
}

impl HMM {
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

    pub fn traverse(&mut self, steps: usize) {
        let rng = rand::thread_rng();
        self.state = HMM::vector_sample(&self.data.prob_start, rng).unwrap();
        for _ in 0..steps {
            let emission = HMM::vector_sample(&self.data.prob_emiss[self.state], rng).unwrap();
            println!("State: {}, Emission: {}", self.state, emission);

            self.state = HMM::vector_sample(&self.data.prob_trans[self.state], rng).unwrap();
        }
    }

    pub fn viterbi(&mut self, observations: Vec<usize>) {
        let mut trellis = Vec::<Vec<TrellisItem>>::with_capacity(observations.len());
        for _ in 0..observations.len() {
            trellis.push(vec![TrellisItem::new(); self.data.count_state]);
        }

        for state in 0..(self.data.count_state) {
            let hop_prob =
                self.data.prob_start[state] * self.data.prob_emiss[state][observations[0]];
            trellis[0][state].max_prob = hop_prob;
            trellis[0][state].cum_prob = hop_prob;
        }

        // Trellis computation
        for time in 1..observations.len() {
            for state_b in 0..(self.data.count_state) {
                for state_a in 0..(self.data.count_state) {
                    let hop_prob = trellis[time - 1][state_a].cum_prob
                        * self.data.prob_trans[state_a][state_b];
                    if hop_prob > trellis[time][state_b].max_prob {
                        trellis[time][state_b].max_path = state_a;
                        trellis[time][state_b].max_prob = hop_prob;
                    }
                    trellis[time][state_b].max_prob *=
                        self.data.prob_emiss[state_b][observations[time]];
                    trellis[time][state_b].cum_prob +=
                        hop_prob * self.data.prob_emiss[state_b][observations[time]];
                }
            }
        }

        // Trellis printing
        println!("Trellis unit (cum_prob, max_prob, max_pointer)");
        for state in 0..self.data.count_state {
            for time in 0..observations.len() {
                print!(
                    "({:.2}, {:.2}, {}) ",
                    trellis[time][state].cum_prob,
                    trellis[time][state].max_prob,
                    trellis[time][state].max_path,
                );
            }
            println!();
        }

        let (max_path_end, max_path_prob) = trellis[observations.len() - 1]
            .iter()
            .enumerate()
            .max_by(|(_, value0), (_, value1)| {
                value0.max_prob.partial_cmp(&value1.max_prob).unwrap()
            })
            .map(|(idx, val)| (idx, val.max_prob))
            .unwrap();

        let cum_path_prob = trellis[observations.len() - 1]
            .iter()
            .max_by(|value0, value1| value0.cum_prob.partial_cmp(&value1.cum_prob).unwrap())
            .map(|val| val.cum_prob)
            .unwrap();

        println!("Cummulative observation probability: {:.4}", cum_path_prob);
        println!("Most probable path probability: {:.4}", max_path_prob);
        print!("Most probable path: ");
        let mut max_path = vec![0; observations.len()];
        max_path[observations.len() - 1] = max_path_end;
        for time in (0..(observations.len() - 1)).rev() {
            max_path[time] = trellis[time + 1][max_path[time + 1]].max_path;
        }
        for time in 0..observations.len() {
            print!("{}-", max_path[time]);
        }
        println!();
    }
}
