use super::GgmlType;
use crate::core::{Result, Tensor, DType};
use crate::core::tensor::Shape;
use std::ffi::c_void;
use std::path::Path;

pub struct GgmlBackend {
    handle: *mut c_void,
    model_path: String,
}

impl GgmlBackend {
    pub fn new() -> Self {
        Self {
            handle: std::ptr::null_mut(),
            model_path: String::new(),
        }
    }
    
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.model_path = path.as_ref().to_string_lossy().to_string();
        Ok(())
    }
    
    pub fn is_loaded(&self) -> bool {
        !self.handle.is_null()
    }
    
    pub fn create_tensor(&self, shape: Shape, dtype: GgmlType) -> Result<Tensor> {
        let _core_dtype = match dtype {
            GgmlType::F32 => DType::F32,
            GgmlType::F16 => DType::F16,
            GgmlType::I32 => DType::I32,
            GgmlType::I16 => DType::I16,
            GgmlType::I8 => DType::I8,
            _ => DType::F32,
        };
        
        Ok(Tensor::zeros(shape))
    }
    
    pub fn compute(&self, _tensor: &mut Tensor) -> Result<()> {
        Ok(())
    }
    
    pub fn synchronize(&self) -> Result<()> {
        Ok(())
    }
    
    pub fn free(&mut self) {
        if !self.handle.is_null() {
            self.handle = std::ptr::null_mut();
        }
    }
}

impl Default for GgmlBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for GgmlBackend {
    fn drop(&mut self) {
        self.free();
    }
}

unsafe impl Send for GgmlBackend {}
unsafe impl Sync for GgmlBackend {}
