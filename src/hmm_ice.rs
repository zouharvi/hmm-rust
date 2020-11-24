use crate::hmm::HMM;

impl HMM {
    #[allow(dead_code)]
    pub fn hmm_ice() -> HMM {
        let mut hmm = HMM::zeroes(2, 3);

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