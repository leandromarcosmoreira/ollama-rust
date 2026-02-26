pub mod ops;

pub use ops::TensorOps;

use crate::core::Result;
use std::ops::{Add, Mul, Sub, Div};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I16,
    I8,
    U8,
}

impl DType {
    pub fn bytes_per_element(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 | DType::I16 => 2,
            DType::I8 | DType::U8 => 1,
        }
    }
    
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16)
    }
    
    pub fn is_int(&self) -> bool {
        matches!(self, DType::I32 | DType::I16 | DType::I8 | DType::U8)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Default)]
pub enum Device {
    #[default]
    Cpu,
    Cuda(usize),
    Metal,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }
    
    pub fn from_slice(dims: &[usize]) -> Self {
        Self { dims: dims.to_vec() }
    }
    
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    
    pub fn len(&self) -> usize {
        self.dims.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.dims.is_empty() || self.dims.iter().all(|&d| d == 0)
    }
    
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
    
    pub fn last(&self) -> Option<&usize> {
        self.dims.last()
    }
    
    pub fn dim(&self, idx: usize) -> Option<usize> {
        self.dims.get(idx).copied()
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
    dtype: DType,
    device: Device,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Shape) -> Self {
        Self {
            data,
            shape,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }
    
    pub fn zeros(shape: Shape) -> Self {
        let numel = shape.numel();
        Self {
            data: vec![0.0; numel],
            shape,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }
    
    pub fn ones(shape: Shape) -> Self {
        let numel = shape.numel();
        Self {
            data: vec![1.0; numel],
            shape,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }
    
    pub fn filled(shape: Shape, value: f32) -> Self {
        let numel = shape.numel();
        Self {
            data: vec![value; numel],
            shape,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }
    
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    
    pub fn data_mut(&mut self) -> &mut Vec<f32> {
        &mut self.data
    }
    
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    
    pub fn device(&self) -> Device {
        self.device
    }
    
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }
    
    pub fn slice(&self, start: usize, end: Option<usize>) -> Result<Self> {
        let end = end.unwrap_or(self.data.len());
        let sliced = self.data[start..end].to_vec();
        
        let mut new_shape = self.shape.clone();
        if let Some(last) = new_shape.dims.last_mut() {
            *last = end - start;
        }
        
        Ok(Self {
            data: sliced,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device,
        })
    }
    
    pub fn reshape(&self, shape: Shape) -> Result<Self> {
        if self.shape.numel() != shape.numel() {
            anyhow::bail!("Cannot reshape: element count mismatch")
        }
        
        Ok(Self {
            data: self.data.clone(),
            shape,
            dtype: self.dtype,
            device: self.device,
        })
    }
    
    pub fn to_dtype(&self, dtype: DType) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype,
            device: self.device,
        }
    }
    
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device,
        }
    }
    pub fn from_candle(t: candle_core::Tensor) -> Result<Self> {
        let shape = Shape::from_slice(t.dims());
        let data = t.flatten_all()?.to_vec1::<f32>()?;
        Ok(Self {
            data,
            shape,
            dtype: DType::F32,
            device: Device::Cpu, // Simplification for now
        })
    }
}

impl Add for Tensor {
    type Output = Tensor;
    
    fn add(self, other: Tensor) -> Self::Output {
        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        
        Tensor {
            data,
            shape: self.shape,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

impl Add<&Tensor> for Tensor {
    type Output = Tensor;
    
    fn add(self, other: &Tensor) -> Self::Output {
        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        
        Tensor {
            data,
            shape: self.shape,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    
    fn sub(self, other: Tensor) -> Self::Output {
        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        
        Tensor {
            data,
            shape: self.shape,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    
    fn mul(self, other: Tensor) -> Self::Output {
        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        
        Tensor {
            data,
            shape: self.shape,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

impl Div for Tensor {
    type Output = Tensor;
    
    fn div(self, other: Tensor) -> Self::Output {
        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a / b)
            .collect();
        
        Tensor {
            data,
            shape: self.shape,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;
    
    fn mul(self, scalar: f32) -> Self::Output {
        let data: Vec<f32> = self.data.iter()
            .map(|&a| a * scalar)
            .collect();
        
        Tensor {
            data,
            shape: self.shape,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_creation() {
        let t = Tensor::zeros(Shape::new(vec![10, 20]));
        assert_eq!(t.shape().dims(), &[10, 20]);
        assert_eq!(t.numel(), 200);
    }
    
    #[test]
    fn test_tensor_add() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Tensor::new(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));
        let c = a + b;
        assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
    }
    
    #[test]
    fn test_tensor_scale() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = a * 2.0;
        assert_eq!(b.data(), &[2.0, 4.0, 6.0]);
    }
}
