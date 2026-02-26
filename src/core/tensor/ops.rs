use super::{Tensor, Shape};
use crate::core::Result;

pub trait TensorOps {
    fn matmul(&self, other: &Tensor) -> Result<Tensor>;
    fn softmax(&self, dim: usize) -> Result<Tensor>;
    fn layer_norm(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor>;
    fn rms_norm(&self, weight: &Tensor, eps: f32) -> Result<Tensor>;
    fn silu(&self) -> Result<Tensor>;
    fn gelu(&self) -> Result<Tensor>;
    fn relu(&self) -> Result<Tensor>;
    fn tanh(&self) -> Result<Tensor>;
    fn sigmoid(&self) -> Result<Tensor>;
    fn transpose(&self, dim1: usize, dim2: usize) -> Result<Tensor>;
    fn permute(&self, dims: &[usize]) -> Result<Tensor>;
    fn contiguous(&self) -> Tensor;
    fn sum(&self, dim: Option<usize>) -> f32;
    fn mean(&self, dim: Option<usize>) -> f32;
    fn max(&self) -> f32;
    fn min(&self) -> f32;
    fn argmax(&self, dim: Option<usize>) -> Vec<usize>;
}

impl TensorOps for Tensor {
    fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let a_dims = self.shape.dims();
        let b_dims = other.shape.dims();
        
        if a_dims.len() < 2 || b_dims.len() < 2 {
            anyhow::bail!("MatMul requires at least 2D tensors");
        }
        
        let m = a_dims[a_dims.len() - 2];
        let k = a_dims[a_dims.len() - 1];
        let k2 = b_dims[b_dims.len() - 2];
        let n = b_dims[b_dims.len() - 1];
        
        if k != k2 {
            anyhow::bail!("MatMul: dimension mismatch ({} != {})", k, k2);
        }
        
        let mut result = vec![0.0; m * n];
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }
        
        Ok(Tensor::new(result, Shape::new(vec![m, n])))
    }
    
    fn softmax(&self, dim: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dim >= dims.len() {
            anyhow::bail!("Softmax: invalid dimension");
        }
        
        let dim_size = dims[dim];
        let outer: usize = dims[..dim].iter().product();
        let inner: usize = dims[dim + 1..].iter().product();
        
        let mut result = self.data.clone();
        
        for o in 0..outer {
            for i in 0..inner {
                let start = o * dim_size * inner + i;
                
                let max = (0..dim_size)
                    .map(|d| self.data[start + d * inner])
                    .fold(f32::NEG_INFINITY, |a, b| a.max(b));
                
                let sum: f32 = (0..dim_size)
                    .map(|d| (self.data[start + d * inner] - max).exp())
                    .sum();
                
                for d in 0..dim_size {
                    result[start + d * inner] = 
                        (self.data[start + d * inner] - max).exp() / sum;
                }
            }
        }
        
        Ok(Tensor {
            data: result,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
        })
    }
    
    fn layer_norm(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
        let n = self.shape.last().copied().unwrap_or(1);
        
        let mean = self.mean(Some(self.shape.len() - 1));
        let var: f32 = self.data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / n as f32;
        
        let std = (var + eps).sqrt();
        
        let data: Vec<f32> = self.data.iter()
            .zip(weight.data.iter())
            .zip(bias.data.iter())
            .map(|((&x, &w), &b)| (x - mean) / std * w + b)
            .collect();
        
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
        })
    }
    
    fn rms_norm(&self, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let n = self.shape.last().copied().unwrap_or(1) as f32;
        
        let ss: f32 = self.data.iter().map(|&x| x * x).sum();
        let rms = (ss / n + eps).sqrt();
        
        let data: Vec<f32> = self.data.iter()
            .zip(weight.data.iter())
            .map(|(&x, &w)| (x / rms) * w)
            .collect();
        
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
        })
    }
    
    fn silu(&self) -> Result<Tensor> {
        let data: Vec<f32> = self.data.iter()
            .map(|&x| x / (1.0 + (-x).exp()))
            .collect();
        
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
        })
    }
    
    fn gelu(&self) -> Result<Tensor> {
        let data: Vec<f32> = self.data.iter()
            .map(|&x| 0.5 * x * (1.0 + (x * 0.797_884_6).tanh()))
            .collect();
        
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
        })
    }
    
    fn relu(&self) -> Result<Tensor> {
        let data: Vec<f32> = self.data.iter()
            .map(|&x| x.max(0.0))
            .collect();
        
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
        })
    }
    
    fn tanh(&self) -> Result<Tensor> {
        let data: Vec<f32> = self.data.iter()
            .map(|&x| x.tanh())
            .collect();
        
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
        })
    }
    
    fn sigmoid(&self) -> Result<Tensor> {
        let data: Vec<f32> = self.data.iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
        })
    }
    
    fn transpose(&self, dim1: usize, dim2: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dim1 >= dims.len() || dim2 >= dims.len() {
            anyhow::bail!("Transpose: invalid dimensions");
        }
        
        let mut new_dims = dims.to_vec();
        new_dims.swap(dim1, dim2);
        
        Ok(Tensor {
            data: self.data.clone(),
            shape: Shape::new(new_dims),
            dtype: self.dtype,
            device: self.device,
        })
    }
    
    fn permute(&self, dims: &[usize]) -> Result<Tensor> {
        let old_dims = self.shape.dims();
        if dims.len() != old_dims.len() {
            anyhow::bail!("Permute: dimension count mismatch");
        }
        
        let new_dims: Vec<usize> = dims.iter()
            .map(|&d| old_dims.get(d).copied().unwrap_or(1))
            .collect();
        
        Ok(Tensor {
            data: self.data.clone(),
            shape: Shape::new(new_dims),
            dtype: self.dtype,
            device: self.device,
        })
    }
    
    fn contiguous(&self) -> Tensor {
        self.clone()
    }
    
    fn sum(&self, dim: Option<usize>) -> f32 {
        match dim {
            Some(_) => self.data.iter().sum(),
            None => self.data.iter().sum(),
        }
    }
    
    fn mean(&self, dim: Option<usize>) -> f32 {
        let count = match dim {
            Some(d) => self.shape.dim(d).unwrap_or(1) as f32,
            None => self.numel() as f32,
        };
        self.sum(dim) / count
    }
    
    fn max(&self) -> f32 {
        self.data.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b))
    }
    
    fn min(&self) -> f32 {
        self.data.iter().cloned().fold(f32::INFINITY, |a, b| a.min(b))
    }
    
    fn argmax(&self, dim: Option<usize>) -> Vec<usize> {
        match dim {
            Some(d) => {
                let dims = self.shape.dims();
                let dim_size = dims.get(d).copied().unwrap_or(1);
                (0..dim_size)
                    .filter_map(|i| self.data.get(i).map(|&v| (i, v)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(i, _)| vec![i])
                    .unwrap_or_default()
            }
            None => {
                self.data.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| vec![i])
                    .unwrap_or_default()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_softmax() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let s = t.softmax(0).unwrap();
        
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_silu() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let s = t.silu().unwrap();
        assert_eq!(s.data.len(), 3);
    }
    
    #[test]
    fn test_rms_norm() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let w = Tensor::ones(Shape::new(vec![4]));
        let n = t.rms_norm(&w, 1e-5).unwrap();
        assert_eq!(n.data.len(), 4);
    }
}
