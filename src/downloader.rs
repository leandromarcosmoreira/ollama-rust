#![allow(dead_code)]
#![allow(unused)]
use anyhow::{anyhow, Result};
use futures::StreamExt;
use reqwest::{header, Client};
use sha2::{Digest, Sha256};
use parking_lot::Mutex;
use std::io::{Seek, SeekFrom, Write as _};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use tokio::fs::OpenOptions;

pub struct Downloader {
    client: Client,
    num_threads: usize,
    chunk_size: u64,
    prefer_aria2c: bool,
}

impl Downloader {
    pub fn new(num_threads: usize, chunk_size: u64) -> Self {
        Self {
            client: Client::new(),
            num_threads,
            chunk_size,
            prefer_aria2c: true,
        }
    }

    #[allow(dead_code)]
    pub fn with_aria2c(mut self, prefer: bool) -> Self {
        self.prefer_aria2c = prefer;
        self
    }

    fn aria2c_available() -> bool {
        Command::new("aria2c")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    pub async fn download<F>(
        &self,
        url: &str,
        dest_path: &Path,
        expected_digest: Option<&str>,
        progress_callback: F,
    ) -> Result<()>
    where
        F: FnMut(u64, u64) + Send + 'static,
    {
        if self.prefer_aria2c && Self::aria2c_available() {
            self.download_with_aria2c(url, dest_path, expected_digest, progress_callback).await
        } else {
            self.download_native(url, dest_path, expected_digest, progress_callback).await
        }
    }

    async fn download_with_aria2c<F>(
        &self,
        url: &str,
        dest_path: &Path,
        expected_digest: Option<&str>,
        mut progress_callback: F,
    ) -> Result<()>
    where
        F: FnMut(u64, u64) + Send + 'static,
    {
        let dest_dir = dest_path.parent()
            .ok_or_else(|| anyhow!("Invalid destination path"))?;
        let filename = dest_path.file_name()
            .ok_or_else(|| anyhow!("Invalid filename"))?
            .to_string_lossy()
            .to_string();

        let total_size = self.get_file_size(url).await?;
        
        let mut cmd = Command::new("aria2c");
        cmd.arg("--no-conf")
           .arg("--allow-overwrite=true")
           .arg("--auto-file-renaming=false")
           .arg("--continue=true")
           .arg(format!("--max-connection-per-server={}", self.num_threads))
           .arg(format!("--split={}", self.num_threads))
           .arg("--min-split-size=1M")
           .arg("--file-allocation=none")
           .arg("--summary-interval=1")
           .arg("--console-log-level=warn")
           .arg("-d").arg(dest_dir)
           .arg("-o").arg(&filename)
           .arg(url);

        let output = cmd.output()?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("aria2c failed: {}", stderr));
        }

        progress_callback(total_size, total_size);

        if let Some(expected) = expected_digest {
            self.verify_digest(dest_path, expected).await?;
        }

        Ok(())
    }

    async fn get_file_size(&self, url: &str) -> Result<u64> {
        let res = self.client.head(url).send().await?;
        if !res.status().is_success() {
            return Err(anyhow!("Failed to get file info: {}", res.status()));
        }

        res.headers()
            .get(header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or_else(|| anyhow!("Could not determine file size"))
    }

    async fn download_native<F>(
        &self,
        url: &str,
        dest_path: &Path,
        expected_digest: Option<&str>,
        mut progress_callback: F,
    ) -> Result<()>
    where
        F: FnMut(u64, u64) + Send + 'static,
    {
        let res = self.client.head(url).send().await?;
        if !res.status().is_success() {
            return Err(anyhow!("Failed to get file info: {}", res.status()));
        }

        let total_size = res
            .headers()
            .get(header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or_else(|| anyhow!("Could not determine file size"))?;

        let accepts_ranges = res
            .headers()
            .get(header::ACCEPT_RANGES)
            .map(|v| v == "bytes")
            .unwrap_or(false);

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(dest_path)
            .await?;
        
        file.set_len(total_size).await?;
        let std_file = file.into_std().await;
        let shared_file = Arc::new(Mutex::new(std_file));

        if accepts_ranges && total_size > self.chunk_size {
            let mut chunks = Vec::new();
            let mut start = 0;
            while start < total_size {
                let end = (start + self.chunk_size - 1).min(total_size - 1);
                chunks.push((start, end));
                start += self.chunk_size;
            }

            let completed_size = Arc::new(Mutex::new(0u64));
            let progress_callback = Arc::new(Mutex::new(progress_callback));

            let mut stream = futures::stream::iter(chunks)
                .map(|(start, end)| {
                    let client = self.client.clone();
                    let url = url.to_string();
                    let shared_file = Arc::clone(&shared_file);
                    let completed_size = Arc::clone(&completed_size);
                    let progress_callback = Arc::clone(&progress_callback);

                    async move {
                        let range = format!("bytes={}-{}", start, end);
                        let res = client
                            .get(&url)
                            .header(header::RANGE, range)
                            .send()
                            .await?;

                        let mut body = res.bytes_stream();
                        let mut offset = start;

                        while let Some(item) = body.next().await {
                            let chunk = item?;
                            let size = chunk.len() as u64;
                            
                            {
                                let mut f = shared_file.lock();
                                f.seek(SeekFrom::Start(offset))?;
                                f.write_all(&chunk)?;
                            }
                            
                            offset += size;
                            let mut completed = completed_size.lock();
                            *completed += size;
                            
                            let mut cb = progress_callback.lock();
                            (cb)(*completed, total_size);
                        }
                        Ok::<(), anyhow::Error>(())
                    }
                })
                .buffer_unordered(self.num_threads);

            while let Some(res) = stream.next().await {
                res?;
            }
        } else {
            let res = self.client.get(url).send().await?;
            let mut body = res.bytes_stream();
            let mut completed = 0u64;
            let mut offset = 0;

            while let Some(item) = body.next().await {
                let chunk = item?;
                let size = chunk.len() as u64;

                {
                    let mut f = shared_file.lock();
                    f.seek(SeekFrom::Start(offset))?;
                    f.write_all(&chunk)?;
                }

                offset += size;
                completed += size;
                (progress_callback)(completed, total_size);
            }
        }

        if let Some(expected) = expected_digest {
            self.verify_digest(dest_path, expected).await?;
        }

        Ok(())
    }

    async fn verify_digest(&self, path: &Path, expected: &str) -> Result<()> {
        let path = path.to_owned();
        let expected = expected.to_owned();
        
        tokio::task::spawn_blocking(move || {
            let mut file = std::fs::File::open(&path)?;
            let mut hasher = Sha256::new();
            std::io::copy(&mut file, &mut hasher)?;
            let hash = hasher.finalize();
            let digest = format!("sha256:{:x}", hash);
            
            if digest != expected {
                return Err(anyhow!("Digest mismatch: expected {}, got {}", expected, digest));
            }
            Ok(())
        }).await?
    }
}

#[allow(dead_code)]
pub struct Aria2cDownloader {
    connections: usize,
    continue_download: bool,
    max_tries: usize,
    retry_wait: usize,
    timeout: usize,
}

#[allow(dead_code)]
impl Aria2cDownloader {
    pub fn new() -> Self {
        Self {
            connections: 16,
            continue_download: true,
            max_tries: 3,
            retry_wait: 5,
            timeout: 60,
        }
    }

    pub fn connections(mut self, n: usize) -> Self {
        self.connections = n;
        self
    }

    pub fn max_tries(mut self, n: usize) -> Self {
        self.max_tries = n;
        self
    }

    pub fn timeout(mut self, secs: usize) -> Self {
        self.timeout = secs;
        self
    }

    pub fn is_available() -> bool {
        Command::new("aria2c")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    pub async fn download(&self, url: &str, dest_path: &Path) -> Result<()> {
        if !Self::is_available() {
            return Err(anyhow!("aria2c not found. Please install aria2c for faster downloads."));
        }

        let dest_dir = dest_path.parent()
            .ok_or_else(|| anyhow!("Invalid destination path"))?;
        let filename = dest_path.file_name()
            .ok_or_else(|| anyhow!("Invalid filename"))?
            .to_string_lossy()
            .to_string();

        let status = Command::new("aria2c")
            .arg("--no-conf")
            .arg("--allow-overwrite=true")
            .arg("--auto-file-renaming=false")
            .arg(format!("--continue={}", self.continue_download))
            .arg(format!("--max-connection-per-server={}", self.connections))
            .arg(format!("--split={}", self.connections))
            .arg("--min-split-size=1M")
            .arg("--file-allocation=none")
            .arg(format!("--max-tries={}", self.max_tries))
            .arg(format!("--retry-wait={}", self.retry_wait))
            .arg(format!("--timeout={}", self.timeout))
            .arg("--console-log-level=warn")
            .arg("-d").arg(dest_dir)
            .arg("-o").arg(&filename)
            .arg(url)
            .status()?;

        if !status.success() {
            return Err(anyhow!("aria2c download failed with status: {}", status));
        }

        Ok(())
    }

    pub async fn download_with_progress<F>(&self, url: &str, dest_path: &Path, mut progress: F) -> Result<()>
    where
        F: FnMut(u64, u64) + Send + 'static,
    {
        self.download(url, dest_path).await?;
        if let Ok(metadata) = std::fs::metadata(dest_path) {
            progress(metadata.len(), metadata.len());
        }
        Ok(())
    }
}

impl Default for Aria2cDownloader {
    fn default() -> Self {
        Self::new()
    }
}
