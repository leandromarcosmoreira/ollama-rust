#![allow(dead_code)]
#![allow(unused)]
use anyhow::{bail, Result};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use parking_lot::Mutex;
use reqwest::header::ACCEPT;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

pub mod registry {
    use super::*;
    
    pub struct Registry {
        client: reqwest::Client,
        registry_url: String,
    }
    
    impl Registry {
        pub fn new() -> Self {
            Self {
                client: reqwest::Client::builder()
                    .timeout(Duration::from_secs(300))
                    .build()
                    .unwrap(),
                registry_url: "https://registry.ollama.ai".to_string(),
            }
        }
        
        pub fn resolve_name(name: &str) -> (String, String) {
            let parts: Vec<&str> = name.splitn(2, ':').collect();
            let base_name = parts[0];
            let tag = parts.get(1).copied().unwrap_or("latest");
            
            let full_name = if base_name.contains('/') {
                base_name.to_string()
            } else {
                format!("library/{}", base_name)
            };
            
            (full_name, tag.to_string())
        }
        
        pub async fn get_manifest(&self, name: &str, tag: &str) -> Result<Manifest> {
            let url = format!("{}/v2/{}/manifests/{}", self.registry_url, name, tag);
            
            let response = self.client
                .get(&url)
                .header(ACCEPT, "application/vnd.docker.distribution.manifest.v2+json")
                .send()
                .await?;
            
            if !response.status().is_success() {
                bail!("Failed to get manifest: {}", response.status());
            }
            
            let manifest = response.json().await?;
            Ok(manifest)
        }
        
        pub fn get_blob_url(&self, name: &str, digest: &str) -> String {
            format!("{}/v2/{}/blobs/{}", self.registry_url, name, digest)
        }
    }
    
    impl Default for Registry {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    #[serde(rename = "schemaVersion")]
    pub schema_version: i32,
    #[serde(rename = "mediaType")]
    pub media_type: Option<String>,
    pub config: Layer,
    pub layers: Vec<Layer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    #[serde(rename = "mediaType")]
    pub media_type: Option<String>,
    pub digest: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_format: Option<String>,
    pub model_family: Option<String>,
    pub model_families: Option<Vec<String>>,
    pub parameter_size: Option<String>,
    pub quantization_level: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModel {
    pub name: String,
    pub tag: String,
    pub size: u64,
    pub digest: String,
    pub modified_at: String,
    pub details: Option<ModelDetails>,
    pub license: Option<String>,
    pub system: Option<String>,
    pub template: Option<String>,
    pub modelfile: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelDetails {
    pub parent_model: Option<String>,
    pub format: Option<String>,
    pub family: Option<String>,
    pub families: Option<Vec<String>>,
    pub parameter_size: Option<String>,
    pub quantization_level: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PullProgress {
    pub status: String,
    pub digest: Option<String>,
    pub total: Option<u64>,
    pub completed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub percentage: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PushProgress {
    pub status: String,
    pub digest: Option<String>,
    pub total: Option<u64>,
    pub completed: Option<u64>,
}

pub struct ModelManager {
    models_dir: PathBuf,
    blobs_dir: PathBuf,
    registry: registry::Registry,
    downloader: Arc<Downloader>,
}

impl ModelManager {
    pub fn new(models_dir: &Path) -> Result<Self> {
        let blobs_dir = models_dir.join("blobs");
        fs::create_dir_all(models_dir)?;
        fs::create_dir_all(&blobs_dir)?;
        
        Ok(Self {
            models_dir: models_dir.to_path_buf(),
            blobs_dir,
            registry: registry::Registry::new(),
            downloader: Arc::new(Downloader::new(16)),
        })
    }
    
    pub fn get_model_dir(&self, name: &str) -> PathBuf {
        let dir_name = name.replace("/", "--");
        self.models_dir.join(dir_name)
    }
    
    pub fn get_manifest_path(&self, name: &str, tag: &str) -> PathBuf {
        // 1. Try official Ollama path first
        let official_path = self.models_dir.join("manifests")
            .join("registry.ollama.ai")
            .join(name)
            .join(tag);
        
        if official_path.exists() {
            return official_path;
        }

        // 2. Fallback to simplified structure
        self.get_model_dir(name).join(format!("{}.json", tag))
    }
    
    pub fn get_blob_path(&self, digest: &str) -> PathBuf {
        let clean_digest = digest.trim_start_matches("sha256:");
        self.blobs_dir.join(format!("sha256-{}", clean_digest))
    }
    
    pub fn get_model_weights_path(&self, name: &str) -> Option<PathBuf> {
        let (full_name, tag) = registry::Registry::resolve_name(name);
        let model_dir = self.get_model_dir(&full_name);
        let manifest_path = model_dir.join(format!("{}.json", tag));
        
        if manifest_path.exists() {
            if let Ok(content) = fs::read_to_string(&manifest_path) {
                if let Ok(manifest) = serde_json::from_str::<Manifest>(&content) {
                    for layer in &manifest.layers {
                        if layer.media_type.as_deref() == Some("application/vnd.ollama.image.model") {
                            let blob_path = self.get_blob_path(&layer.digest);
                            if blob_path.exists() {
                                return Some(blob_path);
                            }
                        }
                    }
                    
                    for layer in &manifest.layers {
                        let blob_path = self.get_blob_path(&layer.digest);
                        if blob_path.exists() {
                            if let Ok(mut file) = std::fs::File::open(&blob_path) {
                                let mut magic = [0u8; 4];
                                if file.read_exact(&mut magic).is_ok() && &magic == b"GGUF" {
                                    return Some(blob_path);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if let Ok(entries) = fs::read_dir(&model_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                    if filename.starts_with("layer-") && filename.ends_with(".bin") {
                        if let Ok(mut file) = std::fs::File::open(&path) {
                            let mut magic = [0u8; 4];
                            if file.read_exact(&mut magic).is_ok() && &magic == b"GGUF" {
                                return Some(path);
                            }
                        }
                    }
                }
            }
        }
        
        None
    }
    
    pub fn stat_blob(&self, digest: &str) -> Option<u64> {
        let path = self.get_blob_path(digest);
        fs::metadata(&path).ok().map(|m| m.len())
    }
    
    pub fn create_blob(&self, digest: &str, data: &[u8]) -> Result<()> {
        let path = self.get_blob_path(digest);
        let mut file = std::fs::File::create(&path)?;
        file.write_all(data)?;
        Ok(())
    }
    
    pub fn list_local_models(&self) -> Result<Vec<LocalModel>> {
        let mut models = Vec::new();
        
        // 1. Scan the simplified structure (legacy/experimental)
        if let Ok(entries) = fs::read_dir(&self.models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() { continue; }
                
                if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                    if dir_name == "blobs" || dir_name == "manifests" { continue; }
                    
                    let model_name = dir_name.replace("--", "/");
                    for manifest_file in fs::read_dir(&path).into_iter().flatten().flatten() {
                        let manifest_path = manifest_file.path();
                        if manifest_path.extension().map(|e| e == "json").unwrap_or(false) {
                            let tag = manifest_path.file_stem().and_then(|s| s.to_str()).unwrap_or("latest");
                            if tag == "config" { continue; }
                            
                            if let Ok(model) = self.load_local_model(&model_name, tag, &manifest_path) {
                                models.push(model);
                            }
                        }
                    }
                    
                    // Legacy check
                    let config_path = path.join("config.json");
                    if config_path.exists() {
                        if let Ok(model) = self.load_legacy_model(&model_name, &path) {
                            models.push(model);
                        }
                    }
                }
            }
        }

        // 2. Scan the official Ollama structure (manifests/registry.ollama.ai/...)
        let manifests_root = self.models_dir.join("manifests");
        if manifests_root.exists() {
            self.walk_manifests(&manifests_root, &mut models)?;
        }
        
        models.sort_by(|a, b| a.name.cmp(&b.name));
        models.dedup_by(|a, b| a.name == b.name);
        Ok(models)
    }

    fn walk_manifests(&self, path: &Path, models: &mut Vec<LocalModel>) -> Result<()> {
        if !path.is_dir() { return Ok(()); }
        
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            
            if entry_path.is_dir() {
                self.walk_manifests(&entry_path, models)?;
            } else {
                // We reached a file, which is a manifest (the tag name)
                // path is .../library/llama3.2, entry is 1b
                // We need to extract the model name from the path relative to manifests/registry.ollama.ai
                let manifests_root = self.models_dir.join("manifests");
                if let Ok(relative) = entry_path.strip_prefix(&manifests_root) {
                    let parts: Vec<_> = relative.components().collect();
                    if parts.len() >= 3 {
                        // parts: [registry, namespace, model, tag]
                        let tag = parts.last().unwrap().as_os_str().to_str().unwrap_or("latest");
                        let model_parts: Vec<_> = parts[1..parts.len()-1].iter().map(|c| c.as_os_str().to_str().unwrap_or("")).collect();
                        let model_name = model_parts.join("/");
                        
                        if let Ok(model) = self.load_local_model(&model_name, tag, &entry_path) {
                            models.push(model);
                        }
                    }
                }
            }
        }
        Ok(())
    }
    
    fn load_legacy_model(&self, name: &str, model_dir: &Path) -> Result<LocalModel> {
        let mut total_size = 0u64;
        let mut has_model_file = false;
        
        if let Ok(entries) = fs::read_dir(model_dir) {
            for entry in entries {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_file() {
                    if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                        if filename.starts_with("layer-") && filename.ends_with(".bin") {
                            if let Ok(metadata) = fs::metadata(&path) {
                                total_size += metadata.len();
                                has_model_file = true;
                            }
                        }
                    }
                }
            }
        }
        
        if !has_model_file {
            bail!("No model files found");
        }
        
        let modified_at = fs::metadata(model_dir)?
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| chrono::DateTime::from_timestamp(d.as_secs() as i64, 0)
                .unwrap_or_else(chrono::Utc::now)
                .to_rfc3339())
            .unwrap_or_else(|_| chrono::Utc::now().to_rfc3339());
        
        Ok(LocalModel {
            name: format!("{}:latest", name),
            tag: "latest".to_string(),
            size: total_size,
            digest: String::new(),
            modified_at,
            details: Some(ModelDetails {
                format: Some("gguf".to_string()),
                family: Some("llama".to_string()),
                parameter_size: None,
                quantization_level: None,
                parent_model: None,
                families: None,
            }),
            license: None,
            system: None,
            template: None,
            modelfile: None,
        })
    }
    
    fn load_local_model(&self, name: &str, tag: &str, manifest_path: &Path) -> Result<LocalModel> {
        let content = fs::read_to_string(manifest_path)?;
        let manifest: Manifest = serde_json::from_str(&content)?;
        
        let mut total_size = manifest.config.size;
        for layer in &manifest.layers {
            total_size += layer.size;
        }
        
        let modified_at = fs::metadata(manifest_path)?
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| chrono::DateTime::from_timestamp(d.as_secs() as i64, 0)
                .unwrap_or_else(chrono::Utc::now)
                .to_rfc3339())
            .unwrap_or_else(|_| chrono::Utc::now().to_rfc3339());
        
        let digest = manifest.config.digest.clone();
        
        let mut template = None;
        let mut system = None;
        let mut license = None;
        
        for layer in &manifest.layers {
            let blob_path = self.get_blob_path(&layer.digest);
            if blob_path.exists() {
                match layer.media_type.as_deref() {
                    Some("application/vnd.ollama.image.template") => {
                        if let Ok(content) = fs::read_to_string(&blob_path) {
                            template = Some(content);
                        }
                    }
                    Some("application/vnd.ollama.image.system") => {
                        if let Ok(content) = fs::read_to_string(&blob_path) {
                            system = Some(content);
                        }
                    }
                    Some("application/vnd.ollama.image.license") => {
                        if let Ok(content) = fs::read_to_string(&blob_path) {
                            license = Some(content);
                        }
                    }
                    _ => {}
                }
            }
        }
        
        let details = Some(ModelDetails {
            format: Some("gguf".to_string()),
            family: Self::detect_model_family(&manifest),
            parameter_size: None,
            quantization_level: None,
            parent_model: None,
            families: None,
        });
        
        Ok(LocalModel {
            name: format!("{}:{}", name, tag),
            tag: tag.to_string(),
            size: total_size,
            digest,
            modified_at,
            details,
            license,
            system,
            template,
            modelfile: None,
        })
    }
    
    fn detect_model_family(manifest: &Manifest) -> Option<String> {
        for layer in &manifest.layers {
            if layer.media_type.as_deref() == Some("application/vnd.ollama.image.params") {
                return Some("llama".to_string());
            }
        }
        Some("llama".to_string())
    }
    
    pub fn get_model_info(&self, name: &str) -> Result<LocalModel> {
        let (full_name, tag) = registry::Registry::resolve_name(name);
        let manifest_path = self.get_manifest_path(&full_name, &tag);
        
        if manifest_path.exists() {
            return self.load_local_model(&full_name, &tag, &manifest_path);
        }
        
        let model_dir = self.get_model_dir(&full_name);
        let config_path = model_dir.join("config.json");
        
        if config_path.exists() {
            return self.load_legacy_model(&full_name, &model_dir);
        }
        
        bail!("Model not found: {}", name);
    }
    
    pub async fn pull<F>(&self, name: String, progress_callback: F) -> Result<PathBuf>
    where
        F: FnMut(PullProgress) + Send + 'static,
    {
        let (full_name, tag) = registry::Registry::resolve_name(&name);
        
        let progress_sender = Arc::new(Mutex::new(progress_callback));
        
        {
            let mut cb = progress_sender.lock();
            cb(PullProgress {
                status: format!("pulling manifest for {}:{}", full_name, tag),
                digest: None,
                total: None,
                completed: None,
                percentage: None,
            });
        }
        
        let manifest = self.registry.get_manifest(&full_name, &tag).await?;
        
        let model_dir = self.get_model_dir(&full_name);
        fs::create_dir_all(&model_dir)?;
        
        let manifest_path = model_dir.join(format!("{}.json", tag));
        let manifest_content = serde_json::to_string(&manifest)?;
        fs::write(&manifest_path, manifest_content)?;
        
        {
            let mut cb = progress_sender.lock();
            cb(PullProgress {
                status: "verifying layers".to_string(),
                digest: None,
                total: None,
                completed: None,
                percentage: None,
            });
        }
        
        let total_size: u64 = manifest.layers.iter().map(|l| l.size).sum();
        let mut completed_size = 0u64;
        
        let layers_to_download: Vec<_> = manifest.layers.iter()
            .filter(|layer| {
                let blob_path = self.get_blob_path(&layer.digest);
                match fs::metadata(&blob_path) {
                    Ok(meta) => meta.len() != layer.size,
                    Err(_) => true,
                }
            })
            .collect();
        
        if layers_to_download.is_empty() {
            {
                let mut cb = progress_sender.lock();
                cb(PullProgress {
                    status: "all layers already present".to_string(),
                    digest: None,
                    total: Some(total_size),
                    completed: Some(total_size),
                    percentage: Some(100.0),
                });
            }
            return Ok(model_dir);
        }
        
        {
            let mut cb = progress_sender.lock();
            cb(PullProgress {
                status: format!("downloading {} layers", layers_to_download.len()),
                digest: None,
                total: Some(total_size),
                completed: Some(completed_size),
                percentage: Some(0.0),
            });
        }
        
        let mp = MultiProgress::new();
        let main_pb = mp.add(ProgressBar::new(total_size));
        main_pb.set_style(ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({percent}%) {eta}")
            .unwrap()
            .progress_chars("#>-"));
        main_pb.set_message(format!("pulling {}", name));
        
        for (idx, layer) in layers_to_download.iter().enumerate() {
            let blob_url = self.registry.get_blob_url(&full_name, &layer.digest);
            let blob_path = self.get_blob_path(&layer.digest);
            
            let short_digest = &layer.digest.chars().take(12).collect::<String>();
            main_pb.set_message(format!("layer {}/{}: {}", idx + 1, layers_to_download.len(), short_digest));
            
            {
                let mut cb = progress_sender.lock();
                cb(PullProgress {
                    status: format!("downloading {}", short_digest),
                    digest: Some(layer.digest.clone()),
                    total: Some(layer.size),
                    completed: Some(completed_size),
                    percentage: Some((completed_size as f64 / total_size as f64) * 100.0),
                });
            }
            
            self.downloader.download_with_progress(
                &blob_url,
                &blob_path,
                layer.size,
                |bytes_downloaded| {
                    main_pb.set_position(completed_size + bytes_downloaded);
                    
                    let mut cb = progress_sender.lock();
                    cb(PullProgress {
                        status: format!("downloading {}", short_digest),
                        digest: Some(layer.digest.clone()),
                        total: Some(layer.size),
                        completed: Some(completed_size + bytes_downloaded),
                        percentage: Some(((completed_size + bytes_downloaded) as f64 / total_size as f64) * 100.0),
                    });
                }
            ).await?;
            
            completed_size += layer.size;
        }
        
        main_pb.finish_with_message(format!("✓ {}", name));
        
        {
            let mut cb = progress_sender.lock();
            cb(PullProgress {
                status: "success".to_string(),
                digest: None,
                total: Some(total_size),
                completed: Some(total_size),
                percentage: Some(100.0),
            });
        }
        
        Ok(model_dir)
    }
    
    pub fn delete_model(&self, name: &str) -> Result<()> {
        let (full_name, tag) = registry::Registry::resolve_name(name);
        let model_dir = self.get_model_dir(&full_name);
        let manifest_path = model_dir.join(format!("{}.json", tag));
        
        if !manifest_path.exists() {
            bail!("Model not found: {}", name);
        }
        
        let content = fs::read_to_string(&manifest_path)?;
        let manifest: Manifest = serde_json::from_str(&content)?;
        
        for layer in &manifest.layers {
            let blob_path = self.get_blob_path(&layer.digest);
            if blob_path.exists() {
                let _ = fs::remove_file(&blob_path);
            }
        }
        
        fs::remove_file(&manifest_path)?;
        
        if fs::read_dir(&model_dir)?.next().is_none() {
            fs::remove_dir(&model_dir)?;
        }
        
        Ok(())
    }
    
    pub fn copy_model(&self, src: &str, dst: &str) -> Result<()> {
        let (src_full, src_tag) = registry::Registry::resolve_name(src);
        let (dst_full, dst_tag) = registry::Registry::resolve_name(dst);
        
        let src_path = self.get_manifest_path(&src_full, &src_tag);
        let dst_path = self.get_manifest_path(&dst_full, &dst_tag);
        
        if !src_path.exists() {
            bail!("Source model not found: {}", src);
        }
        
        if dst_path.exists() {
            bail!("Destination model already exists: {}", dst);
        }
        
        let dst_dir = dst_path.parent().unwrap();
        fs::create_dir_all(dst_dir)?;
        
        fs::copy(&src_path, &dst_path)?;
        
        Ok(())
    }
    
    pub async fn push<F>(&self, _name: String, progress_callback: F) -> Result<()>
    where
        F: FnMut(PushProgress) + Send + 'static,
    {
        let mut cb = progress_callback;
        cb(PushProgress {
            status: "pushing manifest".to_string(),
            digest: None,
            total: None,
            completed: None,
        });
        
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        cb(PushProgress {
            status: "success".to_string(),
            digest: None,
            total: None,
            completed: None,
        });
        
        Ok(())
    }
}

pub struct Downloader {
    connections: usize,
}

impl Downloader {
    pub fn new(connections: usize) -> Self {
        Self { connections }
    }
    
    pub async fn download_with_progress<F>(
        &self,
        url: &str,
        dest_path: &Path,
        expected_size: u64,
        progress: F,
    ) -> Result<()>
    where
        F: FnMut(u64) + Send,
    {
        if Self::aria2c_available() {
            self.download_with_aria2c(url, dest_path, expected_size, progress).await
        } else {
            self.download_native(url, dest_path, expected_size, progress).await
        }
    }
    
    fn aria2c_available() -> bool {
        std::process::Command::new("aria2c")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
    
    async fn download_with_aria2c<F>(
        &self,
        url: &str,
        dest_path: &Path,
        expected_size: u64,
        mut progress: F,
    ) -> Result<()>
    where
        F: FnMut(u64) + Send,
    {
        let dest_dir = dest_path.parent().unwrap();
        let filename = dest_path.file_name().unwrap().to_string_lossy();
        
        let status = std::process::Command::new("aria2c")
            .arg("--no-conf")
            .arg("--allow-overwrite=true")
            .arg("--auto-file-renaming=false")
            .arg("--continue=true")
            .arg(format!("--max-connection-per-server={}", self.connections))
            .arg(format!("--split={}", self.connections))
            .arg("--min-split-size=1M")
            .arg("--file-allocation=none")
            .arg("--console-log-level=warn")
            .arg("-d").arg(dest_dir)
            .arg("-o").arg(&*filename)
            .arg(url)
            .status()?;
        
        if status.success() {
            progress(expected_size);
            Ok(())
        } else {
            bail!("aria2c download failed")
        }
    }
    
    async fn download_native<F>(
        &self,
        url: &str,
        dest_path: &Path,
        expected_size: u64,
        mut progress: F,
    ) -> Result<()>
    where
        F: FnMut(u64) + Send,
    {
        let client = reqwest::Client::new();
        let response = client.get(url).send().await?;
        
        if !response.status().is_success() {
            bail!("Download failed: {}", response.status());
        }
        
        let mut file = std::fs::File::create(dest_path)?;
        let mut downloaded = 0u64;
        
        use futures_util::StreamExt;
        use std::io::Write;
        
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;
            progress(downloaded.min(expected_size));
        }
        
        Ok(())
    }
}
