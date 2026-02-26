use crate::core::model::ModelConfig;
use crate::infra::Result;
use std::path::Path;

pub fn convert<P: AsRef<Path>>(_input: P, _output: P, _config: &ModelConfig) -> Result<()> {
    Ok(())
}
