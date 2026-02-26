pub mod gguf {
    use std::collections::HashMap;
    use std::io::{BufReader, Read};
    use std::path::Path;

    #[derive(Debug, Clone)]
    #[allow(non_camel_case_types)]
    #[allow(dead_code)]
    pub enum DataType {
        Float32,
        Float16,
        Q4_0,
        Q4_1,
        Q5_0,
        Q5_1,
        Q8_0,
        Q8_1,
        Q2_K,
        Q3_K,
        Q4_K,
        Q5_K,
        Q6_K,
        Q8_K,
        I8,
        I16,
        I32,
        I64,
        F64,
        Bool,
        String,
        Unknown(u8),
    }

    #[allow(dead_code)]
    impl DataType {
        pub fn from_u8(val: u8) -> Self {
            match val {
                0 => DataType::Float32,
                1 => DataType::Float16,
                2 => DataType::Q4_0,
                3 => DataType::Q4_1,
                4 => DataType::Q5_0,
                5 => DataType::Q5_1,
                6 => DataType::Q8_0,
                7 => DataType::Q8_1,
                8 => DataType::I8,
                9 => DataType::I16,
                10 => DataType::I32,
                11 => DataType::I64,
                12 => DataType::F64,
                13 => DataType::Bool,
                14 => DataType::String,
                15..=19 => DataType::Q2_K, // Q2_K to Q6_K
                _ => DataType::Unknown(val),
            }
        }

        pub fn bytes_per_element(&self) -> u32 {
            match self {
                DataType::Float32 => 4,
                DataType::Float16 => 2,
                DataType::Q4_0 => 18 / 8,
                DataType::Q4_1 => 20 / 8,
                DataType::Q5_0 => 22 / 8,
                DataType::Q5_1 => 24 / 8,
                DataType::Q8_0 => 34 / 8,
                DataType::Q8_1 => 40 / 8,
                DataType::I8 => 1,
                DataType::I16 => 2,
                DataType::I32 => 4,
                DataType::I64 => 8,
                DataType::F64 => 8,
                DataType::Bool => 1,
                DataType::String => 0, // Variable
                DataType::Q2_K => 2,
                DataType::Q3_K => 3,
                DataType::Q4_K => 4,
                DataType::Q5_K => 5,
                DataType::Q6_K => 6,
                DataType::Q8_K => 8,
                DataType::Unknown(_) => 0,
            }
        }
    }

    #[derive(Debug)]
    #[allow(dead_code)]
    pub enum GGUFValue {
        String(String),
        Int(i64),
        UInt(u64),
        Float(f32),
        Float64(f64),
        Bool(bool),
        Array(Vec<GGUFValue>),
    }

    #[allow(dead_code)]
    impl GGUFValue {
        pub fn type_name(&self) -> &str {
            match self {
                GGUFValue::String(_) => "string",
                GGUFValue::Int(_) => "int",
                GGUFValue::UInt(_) => "uint",
                GGUFValue::Float(_) => "float",
                GGUFValue::Float64(_) => "float64",
                GGUFValue::Bool(_) => "bool",
                GGUFValue::Array(_) => "array",
            }
        }
    }

    #[derive(Debug)]
    #[allow(dead_code)]
    pub struct TensorInfo {
        pub name: String,
        pub shape: Vec<u64>,
        pub dtype: DataType,
        pub offset: u64,
        pub n_elements: u64,
    }

    #[allow(dead_code)]
    impl TensorInfo {
        pub fn size(&self) -> u64 {
            let elem_size = self.dtype.bytes_per_element() as u64;
            match self.dtype {
                DataType::Q4_0 | DataType::Q4_1 | DataType::Q5_0 | DataType::Q5_1 
                | DataType::Q8_0 | DataType::Q8_1 | DataType::Q2_K | DataType::Q3_K 
                | DataType::Q4_K | DataType::Q5_K | DataType::Q6_K | DataType::Q8_K => {
                    self.n_elements.div_ceil(2) * elem_size + 2
                }
                _ => self.n_elements * elem_size,
            }
        }
    }

    #[derive(Debug)]
    #[allow(dead_code)]
    pub struct GGUFReader {
        pub version: u32,
        pub tensors: Vec<TensorInfo>,
        pub metadata: HashMap<String, GGUFValue>,
        pub file_size: u64,
    }

    #[allow(dead_code)]
    impl GGUFReader {
        pub fn open(path: &Path) -> std::io::Result<Self> {
            let file = std::fs::File::open(path)?;
            let file_size = file.metadata()?.len();
            let mut reader = BufReader::new(file);

            let mut magic = [0u8; 4];
            reader.read_exact(&mut magic)?;

            if &magic != b"GGUF" {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Invalid GGUF magic number",
                ));
            }

            let mut version_bytes = [0u8; 4];
            reader.read_exact(&mut version_bytes)?;
            let version = u32::from_le_bytes(version_bytes);

            let mut metadata = HashMap::new();
            let mut tensors = Vec::new();

            let metadata_kv_count = read_u64(&mut reader)? as usize;

            for _ in 0..metadata_kv_count {
                let key = read_string(&mut reader)?;
                let value_type = reader.read_u8()?;
                let value = read_value(&mut reader, value_type)?;
                metadata.insert(key, value);
            }

            let tensor_count = read_u64(&mut reader)? as usize;

            for _ in 0..tensor_count {
                let name = read_string(&mut reader)?;
                let n_dims = reader.read_u8()? as usize;
                let mut shape = Vec::with_capacity(n_dims);
                for _ in 0..n_dims {
                    shape.push(read_u64(&mut reader)?);
                }
                let dtype_byte = reader.read_u8()?;
                let dtype = DataType::from_u8(dtype_byte);
                let offset = read_u64(&mut reader)?;

                let n_elements: u64 = shape.iter().product();

                tensors.push(TensorInfo {
                    name,
                    shape,
                    dtype,
                    offset,
                    n_elements,
                });
            }

            Ok(Self {
                version,
                tensors,
                metadata,
                file_size,
            })
        }

        pub fn get_metadata_string(&self, key: &str) -> Option<String> {
            self.metadata.get(key).and_then(|v| match v {
                GGUFValue::String(s) => Some(s.clone()),
                _ => None,
            })
        }

        pub fn get_metadata_int(&self, key: &str) -> Option<i64> {
            self.metadata.get(key).and_then(|v| match v {
                GGUFValue::Int(i) => Some(*i),
                GGUFValue::UInt(i) => Some(*i as i64),
                _ => None,
            })
        }

        pub fn get_metadata_float(&self, key: &str) -> Option<f32> {
            self.metadata.get(key).and_then(|v| match v {
                GGUFValue::Float(f) => Some(*f),
                _ => None,
            })
        }

        pub fn model_family(&self) -> Option<String> {
            self.get_metadata_string("general.architecture")
        }

        pub fn parameter_count(&self) -> Option<i64> {
            self.get_metadata_int("general.parameter_count")
        }

        pub fn context_length(&self) -> Option<i64> {
            self.get_metadata_int(&format!("{}.context_length", self.model_family().unwrap_or_default()))
        }

        pub fn embedding_length(&self) -> Option<i64> {
            self.get_metadata_int(&format!("{}.embedding_length", self.model_family().unwrap_or_default()))
        }
    }

    #[allow(dead_code)]
    fn read_u64<R: Read>(reader: &mut R) -> std::io::Result<u64> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    #[allow(dead_code)]
    fn read_string<R: Read>(reader: &mut R) -> std::io::Result<String> {
        let len = match read_u64(reader)? {
            0 => return Ok(String::new()),
            n => n as usize,
        };
        
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        
        // Handle potential null terminator
        if let Some(pos) = buf.iter().position(|&b| b == 0) {
            buf.truncate(pos);
        }
        
        String::from_utf8(buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    #[allow(dead_code)]
    fn read_value<R: Read>(reader: &mut R, value_type: u8) -> std::io::Result<GGUFValue> {
        match value_type {
            0 => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                Ok(GGUFValue::Float(f32::from_le_bytes(buf)))
            }
            1 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                Ok(GGUFValue::Int(i64::from_le_bytes(buf)))
            }
            2 => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                Ok(GGUFValue::Bool(buf[0] != 0))
            }
            3 => {
                read_string(reader).map(GGUFValue::String)
            }
            4 => {
                let array_type = reader.read_u8()?;
                let count = read_u64(reader)? as usize;
                let mut arr = Vec::with_capacity(count);
                for _ in 0..count {
                    arr.push(read_value(reader, array_type)?);
                }
                Ok(GGUFValue::Array(arr))
            }
            5 => {
                read_u64(reader).map(GGUFValue::UInt)
            }
            6..=15 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                Ok(GGUFValue::Float64(f64::from_le_bytes(buf)))
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unknown GGUF value type: {}", value_type),
            )),
        }
    }

    #[allow(dead_code)]
    pub trait ReadExt {
        fn read_u8(&mut self) -> std::io::Result<u8>;
    }

    impl<R: Read> ReadExt for R {
        fn read_u8(&mut self) -> std::io::Result<u8> {
            let mut buf = [0u8; 1];
            self.read_exact(&mut buf)?;
            Ok(buf[0])
        }
    }
}

pub mod ggml {
    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    pub struct GGMLType {
        pub name: String,
        pub elements: u64,
        pub bytes: u64,
    }

    #[allow(dead_code)]
    pub fn get_type(name: &str) -> Option<GGMLType> {
        match name {
            "F32" => Some(GGMLType { name: "F32".to_string(), elements: 1, bytes: 4 }),
            "F16" => Some(GGMLType { name: "F16".to_string(), elements: 1, bytes: 2 }),
            "Q4_0" => Some(GGMLType { name: "Q4_0".to_string(), elements: 32, bytes: 18 }),
            "Q4_1" => Some(GGMLType { name: "Q4_1".to_string(), elements: 32, bytes: 20 }),
            "Q5_0" => Some(GGMLType { name: "Q5_0".to_string(), elements: 32, bytes: 22 }),
            "Q5_1" => Some(GGMLType { name: "Q5_1".to_string(), elements: 32, bytes: 24 }),
            "Q8_0" => Some(GGMLType { name: "Q8_0".to_string(), elements: 32, bytes: 34 }),
            "Q8_1" => Some(GGMLType { name: "Q8_1".to_string(), elements: 32, bytes: 40 }),
            "Q8_2" => Some(GGMLType { name: "Q8_2".to_string(), elements: 32, bytes: 52 }),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn type_size(name: &str) -> Option<u64> {
        get_type(name).map(|t| t.bytes)
    }

    #[allow(dead_code)]
    pub fn type_elements(name: &str) -> Option<u64> {
        get_type(name).map(|t| t.elements)
    }
}

use std::io::{Read, SeekFrom};

#[allow(dead_code)]
pub struct BufferSeeker {
    buffer: Vec<u8>,
    position: usize,
}

#[allow(dead_code)]
impl BufferSeeker {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            position: 0,
        }
    }

    pub fn from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        Ok(Self { buffer, position: 0 })
    }

    pub fn read(&mut self, size: usize) -> std::io::Result<&[u8]> {
        let start = self.position;
        let end = (self.position + size).min(self.buffer.len());
        self.position = end;
        Ok(&self.buffer[start..end])
    }

    pub fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match pos {
            SeekFrom::Start(p) => {
                self.position = p as usize;
                Ok(p)
            }
            SeekFrom::End(p) => {
                let pos = (self.buffer.len() as i64 + p) as u64;
                self.position = pos as usize;
                Ok(pos)
            }
            SeekFrom::Current(p) => {
                let pos = (self.position as i64 + p) as u64;
                self.position = pos as usize;
                Ok(pos)
            }
        }
    }
}

impl Default for BufferSeeker {
    fn default() -> Self {
        Self::new()
    }
}
