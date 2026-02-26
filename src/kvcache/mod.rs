use crate::model::Tensor;
use anyhow::Result;
use std::collections::HashMap;

pub trait Cache: Send + Sync {
    fn set_layer(&mut self, layer: usize);
    fn start_forward(&mut self, positions: &[i32], sequences: &[i32]) -> Result<()>;
    fn update(&mut self, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)>;
    fn clear(&mut self);
}

pub struct CausalCache {
    layer: usize,
    cache_len: usize,
    kv_cache: HashMap<usize, (Tensor, Tensor)>,
}

impl CausalCache {
    pub fn new() -> Self {
        Self {
            layer: 0,
            cache_len: 0,
            kv_cache: HashMap::new(),
        }
    }

    pub fn with_len(cache_len: usize) -> Self {
        Self {
            layer: 0,
            cache_len,
            kv_cache: HashMap::new(),
        }
    }
}

impl Cache for CausalCache {
    fn set_layer(&mut self, layer: usize) {
        self.layer = layer;
    }

    fn start_forward(&mut self, _positions: &[i32], _sequences: &[i32]) -> Result<()> {
        Ok(())
    }

    fn update(&mut self, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        if let Some(cached) = self.kv_cache.get(&self.layer) {
            Ok((cached.0.clone(), cached.1.clone()))
        } else {
            self.kv_cache.insert(self.layer, (key.clone(), value.clone()));
            Ok((key.clone(), value.clone()))
        }
    }

    fn clear(&mut self) {
        self.kv_cache.clear();
    }
}

impl Default for CausalCache {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SWACache {
    sliding_window: usize,
    layer: usize,
    kv_cache: HashMap<usize, (Tensor, Tensor)>,
}

impl SWACache {
    pub fn new(sliding_window: usize) -> Self {
        Self {
            sliding_window,
            layer: 0,
            kv_cache: HashMap::new(),
        }
    }
}

impl Cache for SWACache {
    fn set_layer(&mut self, layer: usize) {
        self.layer = layer;
    }

    fn start_forward(&mut self, _positions: &[i32], _sequences: &[i32]) -> Result<()> {
        Ok(())
    }

    fn update(&mut self, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        self.kv_cache.insert(self.layer, (key.clone(), value.clone()));
        Ok((key.clone(), value.clone()))
    }

    fn clear(&mut self) {
        self.kv_cache.clear();
    }
}

pub struct ChunkedAttentionCache {
    chunk_size: usize,
    layer: usize,
    chunks: Vec<Vec<(Tensor, Tensor)>>,
}

impl ChunkedAttentionCache {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            layer: 0,
            chunks: Vec::new(),
        }
    }
}

impl Cache for ChunkedAttentionCache {
    fn set_layer(&mut self, layer: usize) {
        self.layer = layer;
        while self.chunks.len() <= layer {
            self.chunks.push(Vec::new());
        }
    }

    fn start_forward(&mut self, _positions: &[i32], _sequences: &[i32]) -> Result<()> {
        Ok(())
    }

    fn update(&mut self, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        if let Some(chunks) = self.chunks.get_mut(self.layer) {
            chunks.push((key.clone(), value.clone()));
        }
        Ok((key.clone(), value.clone()))
    }

    fn clear(&mut self) {
        self.chunks.clear();
    }
}

pub struct WrapperCache {
    swa_cache: SWACache,
    causal_cache: CausalCache,
}

impl WrapperCache {
    pub fn new(swa_window: usize) -> Self {
        Self {
            swa_cache: SWACache::new(swa_window),
            causal_cache: CausalCache::new(),
        }
    }
}

impl Cache for WrapperCache {
    fn set_layer(&mut self, layer: usize) {
        self.swa_cache.set_layer(layer);
        self.causal_cache.set_layer(layer);
    }

    fn start_forward(&mut self, positions: &[i32], sequences: &[i32]) -> Result<()> {
        self.swa_cache.start_forward(positions, sequences)?;
        self.causal_cache.start_forward(positions, sequences)
    }

    fn update(&mut self, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        self.causal_cache.update(key, value)
    }

    fn clear(&mut self) {
        self.swa_cache.clear();
        self.causal_cache.clear();
    }
}
