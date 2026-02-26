pub mod traits;

pub use traits::{KVCache, CacheEntry, CacheKey};

use crate::core::{Result, Tensor};

pub struct CausalKVCache {
    #[allow(dead_code)]
    layer_count: usize,
    #[allow(dead_code)]
    head_count: usize,
    #[allow(dead_code)]
    head_dim: usize,
    max_seq_len: usize,
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    seq_len: usize,
}

impl CausalKVCache {
    pub fn new(layer_count: usize, head_count: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            layer_count,
            head_count,
            head_dim,
            max_seq_len,
            keys: Vec::with_capacity(layer_count),
            values: Vec::with_capacity(layer_count),
            seq_len: 0,
        }
    }
    
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.seq_len = 0;
    }
    
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
    
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }
}

impl KVCache for CausalKVCache {
    fn update(&mut self, layer: usize, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        if layer >= self.keys.len() {
            self.keys.push(key.clone());
            self.values.push(value.clone());
        } else {
            self.keys[layer] = key.clone();
            self.values[layer] = value.clone();
        }
        
        self.seq_len = self.seq_len.max(key.shape().last().copied().unwrap_or(0));
        
        Ok((key.clone(), value.clone()))
    }
    
    fn get(&self, layer: usize) -> Option<CacheEntry> {
        if layer < self.keys.len() {
            Some(CacheEntry {
                key: self.keys.get(layer).cloned()?,
                value: self.values.get(layer).cloned()?,
            })
        } else {
            None
        }
    }
    
    fn len(&self) -> usize {
        self.seq_len
    }
    
    fn capacity(&self) -> usize {
        self.max_seq_len
    }
}

pub struct SlidingWindowCache {
    inner: CausalKVCache,
    window_size: usize,
}

impl SlidingWindowCache {
    pub fn new(
        layer_count: usize,
        head_count: usize,
        head_dim: usize,
        max_seq_len: usize,
        window_size: usize,
    ) -> Self {
        Self {
            inner: CausalKVCache::new(layer_count, head_count, head_dim, max_seq_len),
            window_size,
        }
    }
}

impl KVCache for SlidingWindowCache {
    fn update(&mut self, layer: usize, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        let start = self.inner.seq_len.saturating_sub(self.window_size);
        if start > 0 {
            let key_sliced = key.slice(start, None)?;
            let value_sliced = value.slice(start, None)?;
            self.inner.update(layer, &key_sliced, &value_sliced)
        } else {
            self.inner.update(layer, key, value)
        }
    }
    
    fn get(&self, layer: usize) -> Option<CacheEntry> {
        self.inner.get(layer)
    }
    
    fn len(&self) -> usize {
        self.inner.len()
    }
    
    fn capacity(&self) -> usize {
        self.inner.capacity().min(self.window_size)
    }
}

pub struct ChunkedCache {
    chunks: Vec<CausalKVCache>,
    chunk_size: usize,
}

impl ChunkedCache {
    pub fn new(
        layer_count: usize,
        head_count: usize,
        head_dim: usize,
        max_seq_len: usize,
        chunk_size: usize,
    ) -> Self {
        let num_chunks = max_seq_len.div_ceil(chunk_size);
        let chunks = (0..num_chunks)
            .map(|_| CausalKVCache::new(layer_count, head_count, head_dim, chunk_size))
            .collect();
        
        Self { chunks, chunk_size }
    }
    
    fn chunk_for_pos(&self, pos: usize) -> usize {
        pos / self.chunk_size
    }
}

impl KVCache for ChunkedCache {
    fn update(&mut self, layer: usize, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        let chunk_idx = self.chunk_for_pos(self.len());
        if chunk_idx < self.chunks.len() {
            self.chunks[chunk_idx].update(layer, key, value)
        } else {
            anyhow::bail!("Cache capacity exceeded")
        }
    }
    
    fn get(&self, layer: usize) -> Option<CacheEntry> {
        for chunk in &self.chunks {
            if let Some(entry) = chunk.get(layer) {
                return Some(entry);
            }
        }
        None
    }
    
    fn len(&self) -> usize {
        self.chunks.iter().map(|c| c.len()).sum()
    }
    
    fn capacity(&self) -> usize {
        self.chunks.len() * self.chunk_size
    }
}
