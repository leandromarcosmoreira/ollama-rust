use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Header {
    #[serde(flatten)]
    pub tensors: HashMap<String, TensorInfo>,
    #[serde(rename = "__metadata__")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

pub struct SafeTensors {
    pub header: Header,
    pub data_offset: u64,
    file_path: std::path::PathBuf,
}

impl SafeTensors {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(&path)?;
        
        let mut header_len_buf = [0u8; 8];
        file.read_exact(&mut header_len_buf)?;
        let header_len = u64::from_le_bytes(header_len_buf);
        
        if header_len > 100 * 1024 * 1024 {
            bail!("Safetensors header too large: {} bytes", header_len);
        }
        
        let mut header_buf = vec![0u8; header_len as usize];
        file.read_exact(&mut header_buf)?;
        
        let header: Header = serde_json::from_slice(&header_buf)?;
        let data_offset = 8 + header_len;
        
        Ok(Self {
            header,
            data_offset,
            file_path: path.as_ref().to_path_buf(),
        })
    }
    
    pub fn get_tensor_data(&self, name: &str) -> Result<Vec<u8>> {
        let info = self.header.tensors.get(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found", name))?;
        
        let mut file = File::open(&self.file_path)?;
        let start = self.data_offset + info.data_offsets[0] as u64;
        let end = self.data_offset + info.data_offsets[1] as u64;
        let len = (end - start) as usize;
        
        file.seek(SeekFrom::Start(start))?;
        let mut data = vec![0u8; len];
        file.read_exact(&mut data)?;
        
        Ok(data)
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.header.tensors.keys().map(|s| s.as_str()).collect()
    }

    pub fn tensor_count(&self) -> usize {
        self.header.tensors.len()
    }

    pub fn total_size(&self) -> usize {
        self.header.tensors.values()
            .map(|t| t.data_offsets[1] - t.data_offsets[0])
            .sum()
    }

    pub fn metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.header.metadata.as_ref()?.get(key)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata("architecture")
            .and_then(|v| v.as_str())
    }

    pub fn dtype_size(dtype: &str) -> usize {
        match dtype {
            "F64" => 8,
            "F32" | "I32" | "U32" => 4,
            "BF16" | "F16" | "I16" | "U16" => 2,
            "U8" | "I8" => 1,
            _ => 2,
        }
    }

    pub fn element_count(shape: &[usize]) -> usize {
        shape.iter().product()
    }
}
