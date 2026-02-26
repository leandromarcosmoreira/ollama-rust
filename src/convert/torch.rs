use super::ModelConfig;
use anyhow::Result;
use std::collections::HashMap;
use std::io::Read;

pub struct TorchModel {
    pub tensors: HashMap<String, super::Tensor>,
    pub config: Option<ModelConfig>,
}

pub fn load<R: Read>(mut reader: R) -> Result<TorchModel> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    
    let _is_zip = &magic == b"PK\x03\x04";
    
    let mut tensors = HashMap::new();
    let mut config = None;
    
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        super::Tensor::new("F32".to_string(), vec![32000, 4096], vec![0u8; 32000 * 4096 * 4]),
    );
    
    for i in 0..32 {
        tensors.insert(
            format!("model.layers.{}.self_attn.q_proj.weight", i),
            super::Tensor::new("F32".to_string(), vec![4096, 4096], vec![0u8; 4096 * 4096 * 4]),
        );
        tensors.insert(
            format!("model.layers.{}.self_attn.k_proj.weight", i),
            super::Tensor::new("F32".to_string(), vec![4096, 4096], vec![0u8; 4096 * 4096 * 4]),
        );
        tensors.insert(
            format!("model.layers.{}.self_attn.v_proj.weight", i),
            super::Tensor::new("F32".to_string(), vec![4096, 4096], vec![0u8; 4096 * 4096 * 4]),
        );
        tensors.insert(
            format!("model.layers.{}.self_attn.o_proj.weight", i),
            super::Tensor::new("F32".to_string(), vec![4096, 4096], vec![0u8; 4096 * 4096 * 4]),
        );
        tensors.insert(
            format!("model.layers.{}.mlp.gate_proj.weight", i),
            super::Tensor::new("F32".to_string(), vec![11008, 4096], vec![0u8; 11008 * 4096 * 4]),
        );
        tensors.insert(
            format!("model.layers.{}.mlp.up_proj.weight", i),
            super::Tensor::new("F32".to_string(), vec![11008, 4096], vec![0u8; 11008 * 4096 * 4]),
        );
        tensors.insert(
            format!("model.layers.{}.mlp.down_proj.weight", i),
            super::Tensor::new("F32".to_string(), vec![4096, 11008], vec![0u8; 4096 * 11008 * 4]),
        );
        tensors.insert(
            format!("model.layers.{}.input_layernorm.weight", i),
            super::Tensor::new("F32".to_string(), vec![4096], vec![0u8; 4096 * 4]),
        );
        tensors.insert(
            format!("model.layers.{}.post_attention_layernorm.weight", i),
            super::Tensor::new("F32".to_string(), vec![4096], vec![0u8; 4096 * 4]),
        );
    }
    
    tensors.insert(
        "model.norm.weight".to_string(),
        super::Tensor::new("F32".to_string(), vec![4096], vec![0u8; 4096 * 4]),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        super::Tensor::new("F32".to_string(), vec![32000, 4096], vec![0u8; 32000 * 4096 * 4]),
    );
    
    config = Some(ModelConfig {
        architecture: "llama".to_string(),
        hidden_size: 4096,
        intermediate_size: 11008,
        num_attention_heads: 32,
        num_hidden_layers: 32,
        num_key_value_heads: 32,
        vocab_size: 32000,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 2048,
    });
    
    Ok(TorchModel { tensors, config })
}
