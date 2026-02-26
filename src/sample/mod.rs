#![allow(clippy::module_inception)]
#![allow(unused)]
pub mod sample {
    use rand::Rng;
    
    #[allow(dead_code)]
    pub struct Sampler {
        temperature: f32,
        top_p: f32,
        top_k: i32,
        repeat_penalty: f32,
    }
    
    impl Default for Sampler {
        fn default() -> Self {
            Self::new()
        }
    }
    
    #[allow(dead_code)]
    impl Sampler {
        pub fn new() -> Self {
            Self {
                temperature: 0.8,
                top_p: 0.9,
                top_k: 40,
                repeat_penalty: 1.1,
            }
        }
        
        pub fn sample(&self, logits: &[f32]) -> usize {
            let mut rng = rand::thread_rng();
            
            let mut candidates: Vec<(usize, f32)> = logits.iter()
                .enumerate()
                .map(|(i, &l)| (i, l))
                .collect();
            
            // Apply temperature
            for (_, logit) in &mut candidates {
                *logit /= self.temperature;
            }
            
            // Apply top-k
            if self.top_k > 0 {
                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                candidates.truncate(self.top_k as usize);
            }
            
            // Apply top-p (nucleus sampling)
            if self.top_p < 1.0 {
                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let mut sum = 0.0f32;
                let cutoff = candidates.first().map(|(_, l)| l).unwrap_or(&0.0) * self.top_p;
                
                candidates.retain(|(_, l)| {
                    if sum < self.top_p || *l > cutoff {
                        sum += l.exp();
                        true
                    } else {
                        false
                    }
                });
            }
            
            // Sample from distribution
            let sum: f32 = candidates.iter().map(|(_, l)| l.exp()).sum();
            let r: f32 = rng.gen();
            let mut cumulative = 0.0;
            
            for (idx, (_, logit)) in candidates.iter().enumerate() {
                cumulative += logit.exp() / sum;
                if cumulative >= r {
                    return idx;
                }
            }
            
            candidates.last().map(|(i, _)| *i).unwrap_or(0)
        }
    }
    
    #[allow(dead_code)]
    pub struct Transforms;
    
    #[allow(dead_code)]
    impl Transforms {
        pub fn apply(json_output: &str) -> String {
            // Simple JSON formatter/cleaner
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_output) {
                serde_json::to_string_pretty(&val).unwrap_or_else(|_| json_output.to_string())
            } else {
                json_output.to_string()
            }
        }
    }
}
