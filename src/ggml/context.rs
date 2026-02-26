use crate::ggml::GGML_MAX_DIMS;
use std::ffi::c_void;
use std::ptr;

pub struct Context {
    ptr: *mut c_void,
}

impl Context {
    pub fn new(mem_size: usize) -> Result<Self, String> {
        let ptr = unsafe { ggml_init(GgmlInitParams {
            mem_size,
            no_alloc: false,
            ctx: ptr::null_mut(),
        }) };
        
        if ptr.is_null() {
            return Err("Failed to create GGML context".to_string());
        }
        
        Ok(Self { ptr })
    }
    
    pub fn new_no_alloc(mem_size: usize) -> Result<Self, String> {
        let ptr = unsafe { ggml_init(GgmlInitParams {
            mem_size,
            no_alloc: true,
            ctx: ptr::null_mut(),
        }) };
        
        if ptr.is_null() {
            return Err("Failed to create GGML context".to_string());
        }
        
        Ok(Self { ptr })
    }
    
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
    
    pub fn new_tensor(&self, ggml_type: u32, dims: &[i64]) -> *mut c_void {
        let n_dims = dims.len() as i32;
        let dims_ptr = dims.as_ptr();
        unsafe { ggml_new_tensor(self.ptr, ggml_type, n_dims, dims_ptr) }
    }
    
    pub fn new_tensor_1d(&self, ggml_type: u32, ne0: i64) -> *mut c_void {
        unsafe { ggml_new_tensor_1d(self.ptr, ggml_type, ne0) }
    }
    
    pub fn new_tensor_2d(&self, ggml_type: u32, ne0: i64, ne1: i64) -> *mut c_void {
        unsafe { ggml_new_tensor_2d(self.ptr, ggml_type, ne0, ne1) }
    }
    
    pub fn new_tensor_3d(&self, ggml_type: u32, ne0: i64, ne1: i64, ne2: i64) -> *mut c_void {
        unsafe { ggml_new_tensor_3d(self.ptr, ggml_type, ne0, ne1, ne2) }
    }
    
    pub fn new_tensor_4d(&self, ggml_type: u32, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> *mut c_void {
        unsafe { ggml_new_tensor_4d(self.ptr, ggml_type, ne0, ne1, ne2, ne3) }
    }
    
    pub fn get_first_tensor(&self) -> *mut c_void {
        unsafe { ggml_get_first_tensor(self.ptr) }
    }
    
    pub fn get_next_tensor(&self, t: *mut c_void) -> *mut c_void {
        unsafe { ggml_get_next_tensor(self.ptr, t) }
    }
    
    pub fn get_tensor(&self, name: &str) -> Option<*mut c_void> {
        let c_name = std::ffi::CString::new(name).ok()?;
        let t = unsafe { ggml_get_tensor(self.ptr, c_name.as_ptr()) };
        if t.is_null() { None } else { Some(t) }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ggml_free(self.ptr);
            }
        }
    }
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

#[repr(C)]
struct GgmlInitParams {
    mem_size: usize,
    no_alloc: bool,
    ctx: *mut c_void,
}

extern "C" {
    fn ggml_init(params: GgmlInitParams) -> *mut c_void;
    fn ggml_free(ctx: *mut c_void);
    fn ggml_new_tensor(ctx: *mut c_void, ggml_type: u32, n_dims: i32, ne: *const i64) -> *mut c_void;
    fn ggml_new_tensor_1d(ctx: *mut c_void, ggml_type: u32, ne0: i64) -> *mut c_void;
    fn ggml_new_tensor_2d(ctx: *mut c_void, ggml_type: u32, ne0: i64, ne1: i64) -> *mut c_void;
    fn ggml_new_tensor_3d(ctx: *mut c_void, ggml_type: u32, ne0: i64, ne1: i64, ne2: i64) -> *mut c_void;
    fn ggml_new_tensor_4d(ctx: *mut c_void, ggml_type: u32, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> *mut c_void;
    fn ggml_get_first_tensor(ctx: *mut c_void) -> *mut c_void;
    fn ggml_get_next_tensor(ctx: *mut c_void, t: *mut c_void) -> *mut c_void;
    fn ggml_get_tensor(ctx: *mut c_void, name: *const std::os::raw::c_char) -> *mut c_void;
}
