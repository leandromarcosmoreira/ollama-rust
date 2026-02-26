use super::GgmlBackend;
use crate::core::{Result, Tensor};
use crate::core::tensor::Shape;

pub struct GgmlContext {
    #[allow(dead_code)]
    backend: GgmlBackend,
}

impl GgmlContext {
    pub fn new(backend: GgmlBackend) -> Self {
        Self { backend }
    }
    
    pub fn tensor_zeros(&self, shape: Shape) -> Result<Tensor> {
        Ok(Tensor::zeros(shape))
    }
    
    pub fn tensor_ones(&self, shape: Shape) -> Result<Tensor> {
        Ok(Tensor::ones(shape))
    }
    
    pub fn tensor_from_data(&self, data: Vec<f32>, shape: Shape) -> Result<Tensor> {
        Ok(Tensor::new(data, shape))
    }
}
