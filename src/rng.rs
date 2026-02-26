pub struct SeededRng {
    state: u64,
}

impl SeededRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn gen_range(&mut self, range: std::ops::Range<f64>) -> f64 {
        self.next_u64();
        let normalized = (self.state as f64) / (u64::MAX as f64);
        range.start + normalized * (range.end - range.start)
    }

    fn next_u64(&mut self) {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
    }
}
