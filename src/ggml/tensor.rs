use crate::ggml::{GgmlType, GGML_MAX_DIMS};
use std::ffi::{c_void, CStr, CString};
use std::sync::Arc;

pub struct GgmlTensor {
    ptr: *mut c_void,
    owned: bool,
}

impl GgmlTensor {
    pub fn new(ptr: *mut c_void) -> Self {
        Self { ptr, owned: false }
    }
    
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
    
    pub fn dims(&self) -> i32 {
        unsafe { ggml_n_dims(self.ptr) }
    }
    
    pub fn shape(&self) -> Vec<i64> {
        let n_dims = self.dims();
        let mut shape = Vec::with_capacity(n_dims as usize);
        unsafe {
            for i in 0..n_dims as usize {
                shape.push(ggml_ne(self.ptr, i));
            }
        }
        shape
    }
    
    pub fn dim(&self, axis: usize) -> i64 {
        unsafe { ggml_ne(self.ptr, axis) }
    }
    
    pub fn stride(&self, axis: usize) -> i64 {
        unsafe { ggml_nb(self.ptr, axis) }
    }
    
    pub fn nbytes(&self) -> usize {
        unsafe { ggml_nbytes(self.ptr) }
    }
    
    pub fn nelements(&self) -> usize {
        unsafe { ggml_nelements(self.ptr) as usize }
    }
    
    pub fn ggml_type(&self) -> GgmlType {
        let t = unsafe { ggml_type(self.ptr) };
        match t {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            16 => GgmlType::I8,
            17 => GgmlType::I16,
            18 => GgmlType::I32,
            20 => GgmlType::F64,
            21 => GgmlType::BF16,
            _ => GgmlType::F32,
        }
    }
    
    pub fn name(&self) -> String {
        unsafe {
            let name_ptr = ggml_get_name(self.ptr);
            if name_ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(name_ptr).to_string_lossy().to_string()
            }
        }
    }
    
    pub fn set_name(&mut self, name: &str) {
        if let Ok(c_name) = CString::new(name) {
            unsafe {
                ggml_set_name(self.ptr, c_name.as_ptr());
            }
        }
    }
    
    pub fn is_contiguous(&self) -> bool {
        unsafe { ggml_is_contiguous(self.ptr) }
    }
    
    pub fn is_quantized(&self) -> bool {
        unsafe { ggml_is_quantized(self.ptr) }
    }
    
    pub fn get_data(&self) -> Vec<u8> {
        let size = self.nbytes();
        if size == 0 {
            return Vec::new();
        }
        
        let mut data = vec![0u8; size];
        unsafe {
            ggml_backend_tensor_get(self.ptr, data.as_mut_ptr() as *mut c_void, 0, size);
        }
        data
    }
    
    pub fn get_floats(&self) -> Vec<f32> {
        let n = self.nelements();
        if n == 0 {
            return Vec::new();
        }
        
        let mut data = vec![0.0f32; n];
        unsafe {
            ggml_backend_tensor_get(self.ptr, data.as_mut_ptr() as *mut c_void, 0, n * 4);
        }
        data
    }
    
    pub fn set_data(&self, data: &[u8]) {
        if !data.is_empty() {
            unsafe {
                ggml_backend_tensor_set(self.ptr, data.as_ptr() as *const c_void, 0, data.len());
            }
        }
    }
    
    pub fn set_floats(&self, data: &[f32]) {
        if !data.is_empty() {
            unsafe {
                let bytes = std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4);
                ggml_backend_tensor_set(self.ptr, bytes.as_ptr() as *const c_void, 0, bytes.len());
            }
        }
    }
    
    pub fn set_zero(&self) {
        unsafe { ggml_set_zero(self.ptr) }
    }
}

impl Clone for GgmlTensor {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            owned: false,
        }
    }
}

unsafe impl Send for GgmlTensor {}
unsafe impl Sync for GgmlTensor {}

extern "C" {
    fn ggml_n_dims(t: *mut c_void) -> i32;
    fn ggml_ne(t: *mut c_void, i: usize) -> i64;
    fn ggml_nb(t: *mut c_void, i: usize) -> i64;
    fn ggml_nbytes(t: *mut c_void) -> usize;
    fn ggml_nelements(t: *mut c_void) -> i64;
    fn ggml_type(t: *mut c_void) -> u32;
    fn ggml_get_name(t: *mut c_void) -> *const std::os::raw::c_char;
    fn ggml_set_name(t: *mut c_void, name: *const std::os::raw::c_char);
    fn ggml_is_contiguous(t: *mut c_void) -> bool;
    fn ggml_is_quantized(t: *mut c_void) -> bool;
    fn ggml_set_zero(t: *mut c_void);
    fn ggml_backend_tensor_get(t: *mut c_void, data: *mut c_void, offset: usize, size: usize);
    fn ggml_backend_tensor_set(t: *mut c_void, data: *const c_void, offset: usize, size: usize);
}
