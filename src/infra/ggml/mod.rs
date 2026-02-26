mod backend;
mod context;
mod tensor;

pub use backend::GgmlBackend;
pub use context::GgmlContext;
pub use tensor::GgmlTensor;

pub const GGML_MAX_DIMS: usize = 4;

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
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
}

impl GgmlType {
    pub fn bytes_per_element(&self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::I32 => 4,
            GgmlType::I16 => 2,
            GgmlType::I8 => 1,
            _ => 1,
        }
    }
}
