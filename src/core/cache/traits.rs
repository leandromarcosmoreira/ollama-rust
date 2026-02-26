use crate::core::{Result, Tensor};

pub trait KVCache: Send + Sync {
    fn update(&mut self, layer: usize, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)>;
    fn get(&self, layer: usize) -> Option<CacheEntry>;
    fn len(&self) -> usize;
    fn capacity(&self) -> usize;
    
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub key: Tensor,
    pub value: Tensor,
}

impl CacheEntry {
    pub fn new(key: Tensor, value: Tensor) -> Self {
        Self { key, value }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub layer: usize,
    pub position: usize,
}

impl CacheKey {
    pub fn new(layer: usize, position: usize) -> Self {
        Self { layer, position }
    }
}

pub trait CacheStrategy: Send + Sync {
    fn should_evict(&self, key: CacheKey, current_seq: usize) -> bool;
    fn priority(&self, key: CacheKey) -> f32;
}

pub struct LruStrategy {
    capacity: usize,
}

impl LruStrategy {
    pub fn new(capacity: usize) -> Self {
        Self { capacity }
    }
}

impl CacheStrategy for LruStrategy {
    fn should_evict(&self, _key: CacheKey, current_seq: usize) -> bool {
        current_seq > self.capacity
    }
    
    fn priority(&self, _key: CacheKey) -> f32 {
        0.0
    }
}

pub struct HybridCache<C1, C2> 
where
    C1: KVCache,
    C2: KVCache,
{
    primary: C1,
    secondary: C2,
}

impl<C1, C2> HybridCache<C1, C2>
where
    C1: KVCache,
    C2: KVCache,
{
    pub fn new(primary: C1, secondary: C2) -> Self {
        Self { primary, secondary }
    }
}

impl<C1, C2> KVCache for HybridCache<C1, C2>
where
    C1: KVCache,
    C2: KVCache,
{
    fn update(&mut self, layer: usize, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        self.primary.update(layer, key, value)
    }
    
    fn get(&self, layer: usize) -> Option<CacheEntry> {
        self.primary.get(layer)
            .or_else(|| self.secondary.get(layer))
    }
    
    fn len(&self) -> usize {
        self.primary.len()
    }
    
    fn capacity(&self) -> usize {
        self.primary.capacity() + self.secondary.capacity()
    }
}
