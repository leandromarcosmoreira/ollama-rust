use crate::ggml::{GgmlType, GGML_MAX_DIMS};
use std::ffi::{c_void, CStr, CString};
use std::ptr;

pub struct Backend {
    ptr: *mut c_void,
    model_path: String,
}

impl Backend {
    pub fn new(model_path: &str) -> Result<Self, String> {
        let c_path = CString::new(model_path).map_err(|e| e.to_string())?;
        
        let ptr = unsafe { ggml_backend_new(c_path.as_ptr()) };
        if ptr.is_null() {
            return Err("Failed to create GGML backend".to_string());
        }
        
        Ok(Self {
            ptr,
            model_path: model_path.to_string(),
        })
    }
    
    pub fn cpu() -> Self {
        let ptr = unsafe { ggml_backend_cpu_new() };
        Self {
            ptr,
            model_path: String::new(),
        }
    }
    
    pub fn gpu(device_index: usize) -> Result<Self, String> {
        let ptr = unsafe { ggml_backend_gpu_new(device_index as i32) };
        if ptr.is_null() {
            return Err("Failed to create GPU backend".to_string());
        }
        
        Ok(Self {
            ptr,
            model_path: String::new(),
        })
    }
    
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
    
    pub fn name(&self) -> String {
        unsafe {
            let name_ptr = ggml_backend_name(self.ptr);
            if name_ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(name_ptr).to_string_lossy().to_string()
            }
        }
    }
    
    pub fn get_default_buffer_type(&self) -> *mut c_void {
        unsafe { ggml_backend_get_default_buffer_type(self.ptr) }
    }
    
    pub fn set_n_threads(&mut self, n_threads: i32) {
        unsafe { ggml_backend_cpu_set_n_threads(self.ptr, n_threads) }
    }
    
    pub fn is_cpu(&self) -> bool {
        unsafe { ggml_backend_is_cpu(self.ptr) }
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ggml_backend_free(self.ptr);
            }
        }
    }
}

unsafe impl Send for Backend {}
unsafe impl Sync for Backend {}

extern "C" {
    fn ggml_backend_new(model_path: *const std::os::raw::c_char) -> *mut c_void;
    fn ggml_backend_cpu_new() -> *mut c_void;
    fn ggml_backend_gpu_new(device_index: i32) -> *mut c_void;
    fn ggml_backend_free(backend: *mut c_void);
    fn ggml_backend_name(backend: *mut c_void) -> *const std::os::raw::c_char;
    fn ggml_backend_get_default_buffer_type(backend: *mut c_void) -> *mut c_void;
    fn ggml_backend_cpu_set_n_threads(backend: *mut c_void, n_threads: i32);
    fn ggml_backend_is_cpu(backend: *mut c_void) -> bool;
}
