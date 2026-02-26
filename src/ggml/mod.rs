pub mod backend;
pub mod context;
pub mod tensor;

pub use backend::Backend;
pub use context::Context;
pub use tensor::GgmlTensor;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    BF16 = 21,
}

impl GgmlType {
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 => 1,
            Self::I64 | Self::F64 => 8,
            _ => 1,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendDeviceType {
    CPU = 0,
    GPU = 1,
    IGPU = 2,
    ACCEL = 3,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Success = 0,
    Failed = 1,
    NoMem = 2,
}

pub const GGML_MAX_DIMS: usize = 4;

pub fn init() {
    unsafe {
        ggml_backend_init();
    }
}

pub fn get_device_count() -> usize {
    unsafe { ggml_backend_dev_count() as usize }
}

pub fn get_gpu_devices() -> Vec<DeviceInfo> {
    let mut devices = Vec::new();
    let count = get_device_count();
    
    for i in 0..count {
        unsafe {
            let dev = ggml_backend_dev_get(i as i32);
            let dev_type = ggml_backend_dev_type(dev);
            
            if dev_type == BackendDeviceType::GPU as u32 || dev_type == BackendDeviceType::IGPU as u32 {
                let mut props: GgmlBackendDevProps = std::mem::zeroed();
                ggml_backend_dev_get_props(dev, &mut props);
                
                devices.push(DeviceInfo {
                    index: i,
                    name: if props.name.is_null() { String::new() } else { std::ffi::CStr::from_ptr(props.name).to_string_lossy().to_string() },
                    description: if props.description.is_null() { String::new() } else { std::ffi::CStr::from_ptr(props.description).to_string_lossy().to_string() },
                    memory_total: props.memory_total,
                    memory_free: props.memory_free,
                });
            }
        }
    }
    
    devices
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub description: String,
    pub memory_total: u64,
    pub memory_free: u64,
}

#[repr(C)]
struct GgmlBackendDevProps {
    name: *const std::os::raw::c_char,
    description: *const std::os::raw::c_char,
    id: *const std::os::raw::c_char,
    library: *const std::os::raw::c_char,
    device_id: *const std::os::raw::c_char,
    compute_major: i32,
    compute_minor: i32,
    driver_major: i32,
    driver_minor: i32,
    integrated: bool,
    memory_total: u64,
    memory_free: u64,
}

extern "C" {
    fn ggml_backend_init();
    fn ggml_backend_dev_count() -> i32;
    fn ggml_backend_dev_get(i: i32) -> *mut std::ffi::c_void;
    fn ggml_backend_dev_type(dev: *mut std::ffi::c_void) -> u32;
    fn ggml_backend_dev_get_props(dev: *mut std::ffi::c_void, props: *mut GgmlBackendDevProps);
}
