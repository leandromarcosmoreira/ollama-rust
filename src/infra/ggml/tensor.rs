use crate::core::Tensor;
use crate::core::tensor::Shape;

pub struct GgmlTensor {
    inner: Tensor,
}

impl GgmlTensor {
    pub fn new(tensor: Tensor) -> Self {
        Self { inner: tensor }
    }
    
    pub fn inner(&self) -> &Tensor {
        &self.inner
    }
    
    pub fn into_inner(self) -> Tensor {
        self.inner
    }
    
    pub fn shape(&self) -> &Shape {
        self.inner.shape()
    }
    
    pub fn data(&self) -> &[f32] {
        self.inner.data()
    }
}

impl From<Tensor> for GgmlTensor {
    fn from(tensor: Tensor) -> Self {
        Self::new(tensor)
    }
}

impl From<GgmlTensor> for Tensor {
    fn from(ggml_tensor: GgmlTensor) -> Self {
        ggml_tensor.into_inner()
    }
}
