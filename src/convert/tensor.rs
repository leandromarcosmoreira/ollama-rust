#[derive(Debug, Clone)]
pub struct Tensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn new(dtype: String, shape: Vec<usize>, data: Vec<u8>) -> Self {
        Self { dtype, shape, data }
    }

    pub fn zeros(dtype: &str, shape: Vec<usize>) -> Self {
        let element_size = Self::dtype_size(dtype);
        let total: usize = shape.iter().product();
        let data = vec![0u8; total * element_size];
        Self {
            dtype: dtype.to_string(),
            shape,
            data,
        }
    }

    pub fn dtype_size(dtype: &str) -> usize {
        match dtype {
            "F64" => 8,
            "F32" | "I32" | "U32" => 4,
            "BF16" | "F16" | "I16" | "U16" => 2,
            "U8" | "I8" | "BOOL" => 1,
            "Q4_0" | "Q4_1" => 1,
            "Q8_0" => 1,
            _ => 2,
        }
    }

    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    pub fn as_f32_slice(&self) -> Vec<f32> {
        if self.dtype != "F32" {
            return Vec::new();
        }
        
        let count = self.element_count();
        let mut result = Vec::with_capacity(count);
        
        for chunk in self.data.chunks_exact(4) {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            result.push(f32::from_le_bytes(bytes));
        }
        
        result
    }

    pub fn from_f32_slice(data: &[f32], shape: Vec<usize>) -> Self {
        let bytes: Vec<u8> = data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        
        Self {
            dtype: "F32".to_string(),
            shape,
            data: bytes,
        }
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, &'static str> {
        let old_count: usize = self.shape.iter().product();
        let new_count: usize = new_shape.iter().product();
        
        if old_count != new_count {
            return Err("Shape mismatch in reshape");
        }
        
        Ok(Self {
            dtype: self.dtype.clone(),
            shape: new_shape,
            data: self.data.clone(),
        })
    }

    pub fn transpose(&self) -> Self {
        if self.shape.len() != 2 {
            return self.clone();
        }
        
        let rows = self.shape[0];
        let cols = self.shape[1];
        let new_shape = vec![cols, rows];
        
        if self.dtype != "F32" {
            return Self {
                dtype: self.dtype.clone(),
                shape: new_shape,
                data: self.data.clone(),
            };
        }
        
        let data = self.as_f32_slice();
        let mut transposed = vec![0.0f32; data.len()];
        
        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = data[i * cols + j];
            }
        }
        
        Self::from_f32_slice(&transposed, new_shape)
    }

    pub fn quantize_q4_0(&self) -> Self {
        Self {
            dtype: "Q4_0".to_string(),
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
    }

    pub fn quantize_q8_0(&self) -> Self {
        Self {
            dtype: "Q8_0".to_string(),
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
    }

    pub fn to_f16(&self) -> Self {
        Self {
            dtype: "F16".to_string(),
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TensorData {
    F32(Vec<f32>),
    F16(Vec<u16>),
    BF16(Vec<u16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bytes(Vec<u8>),
}

impl TensorData {
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::F32(v) => v.iter().flat_map(|f| f.to_le_bytes()).collect(),
            Self::F16(v) | Self::BF16(v) => v.iter().flat_map(|f| f.to_le_bytes()).collect(),
            Self::I32(v) => v.iter().flat_map(|f| f.to_le_bytes()).collect(),
            Self::I64(v) => v.iter().flat_map(|f| f.to_le_bytes()).collect(),
            Self::Bytes(v) => v.clone(),
        }
    }
}
