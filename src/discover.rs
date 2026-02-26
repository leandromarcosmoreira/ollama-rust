use std::path::Path;
use std::process::Command;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor: String,
    pub total_vram: u64,
    pub free_vram: u64,
    pub name: String,
    pub compute_capability: Option<String>,
    pub uuid: Option<String>,
    pub driver_version: Option<String>,
    pub cuda_version: Option<String>,
    pub multiprocessors: Option<usize>,
    pub max_clock_mhz: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct CpuInfo {
    pub cores: usize,
    pub total_memory: u64,
    pub free_memory: u64,
    pub model: Option<String>,
    pub frequency_mhz: Option<u32>,
}

#[allow(dead_code)]
pub fn get_cpu_info() -> CpuInfo {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
        
    let mut total_memory = 0u64;
    let mut free_memory = 0u64;
    let mut model = None;
    let mut frequency_mhz = None;

    if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                total_memory = parse_meminfo_line(line);
            } else if line.starts_with("MemAvailable:") {
                free_memory = parse_meminfo_line(line);
            }
        }
    }

    if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in content.lines() {
            if line.starts_with("model name") {
                model = line.split(':').nth(1).map(|s| s.trim().to_string());
            }
            if line.starts_with("cpu MHz") {
                frequency_mhz = line.split(':').nth(1)
                    .and_then(|s| s.trim().parse::<f32>().ok())
                    .map(|f| f as u32);
            }
        }
    }

    CpuInfo {
        cores,
        total_memory,
        free_memory,
        model,
        frequency_mhz,
    }
}

fn parse_meminfo_line(line: &str) -> u64 {
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

pub fn discover_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    
    discover_nvidia_gpus(&mut gpus);
    discover_amd_gpus(&mut gpus);
    discover_intel_gpus(&mut gpus);
    
    gpus
}

fn discover_nvidia_gpus(gpus: &mut Vec<GpuInfo>) {
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free,compute_cap,uuid,driver_version,multiprocessor_count,clocks.max.sm",
            "--format=csv,noheader,nounits"
        ])
        .output() 
    {
        Ok(o) => o,
        Err(_) => return,
    };

    if !output.status.success() {
        return;
    }

    let out_str = String::from_utf8_lossy(&output.stdout);
    
    for line in out_str.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 3 {
            let total = parts[1].parse::<u64>().unwrap_or(0) * 1024 * 1024;
            let free = parts[2].parse::<u64>().unwrap_or(0) * 1024 * 1024;
            
            gpus.push(GpuInfo {
                vendor: "nvidia".to_string(),
                total_vram: total,
                free_vram: free,
                name: parts[0].to_string(),
                compute_capability: parts.get(3).map(|s| s.to_string()),
                uuid: parts.get(4).map(|s| s.to_string()),
                driver_version: parts.get(5).map(|s| s.to_string()),
                cuda_version: get_cuda_version(),
                multiprocessors: parts.get(6).and_then(|s| s.parse().ok()),
                max_clock_mhz: parts.get(7).and_then(|s| s.parse().ok()),
            });
        }
    }
}

fn get_cuda_version() -> Option<String> {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            let out = String::from_utf8_lossy(&o.stdout);
            for line in out.lines() {
                if line.contains("release") {
                    return line.split("release")
                        .nth(1)
                        .and_then(|s| s.split(',').next())
                        .map(|s| s.trim().to_string());
                }
            }
            None
        })
}

fn discover_amd_gpus(gpus: &mut Vec<GpuInfo>) {
    for i in 0..8 {
        let render_node = 128 + i;
        let path = format!("/sys/class/drm/renderD{}/device/mem_info_vram_total", render_node);
        if Path::new(&path).exists() {
            if let Ok(info) = discover_amd_sysfs(render_node) {
                gpus.push(info);
            }
        }
    }
}

fn discover_amd_sysfs(render_node: i32) -> Result<GpuInfo, std::io::Error> {
    let total_path = format!("/sys/class/drm/renderD{}/device/mem_info_vram_total", render_node);
    let used_path = format!("/sys/class/drm/renderD{}/device/mem_info_vram_used", render_node);
    let name_path = format!("/sys/class/drm/renderD{}/device/product_name", render_node);
    
    let total = std::fs::read_to_string(&total_path)?.trim().parse::<u64>().unwrap_or(0);
    let used = std::fs::read_to_string(&used_path)?.trim().parse::<u64>().unwrap_or(0);
    let name = std::fs::read_to_string(&name_path)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| format!("AMD Radeon (renderD{})", render_node));
    
    Ok(GpuInfo {
        vendor: "amd".to_string(),
        total_vram: total,
        free_vram: total.saturating_sub(used),
        name,
        compute_capability: None,
        uuid: None,
        driver_version: get_amd_driver_version(render_node),
        cuda_version: None,
        multiprocessors: None,
        max_clock_mhz: None,
    })
}

fn get_amd_driver_version(render_node: i32) -> Option<String> {
    let version_path = format!("/sys/class/drm/renderD{}/device/driver/module/version", render_node);
    std::fs::read_to_string(version_path)
        .map(|s| s.trim().to_string())
        .ok()
}

fn discover_intel_gpus(gpus: &mut Vec<GpuInfo>) {
    if Path::new("/sys/class/drm/card0/device/gt_cur_freq_mhz").exists() {
        let mut total_memory = 0u64;
        let mut free_memory = 0u64;
        
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    total_memory = parse_meminfo_line(line) / 2;
                } else if line.starts_with("MemAvailable:") {
                    free_memory = parse_meminfo_line(line) / 2;
                }
            }
        }

        let name = std::fs::read_to_string("/sys/class/drm/card0/device/device")
            .map(|s| format!("Intel GPU ({})", s.trim()))
            .unwrap_or_else(|_| "Intel Integrated Graphics".to_string());

        gpus.push(GpuInfo {
            vendor: "intel".to_string(),
            total_vram: total_memory,
            free_vram: free_memory,
            name,
            compute_capability: None,
            uuid: None,
            driver_version: None,
            cuda_version: None,
            multiprocessors: None,
            max_clock_mhz: None,
        });
    }
}

pub fn estimate_gpu_layers(model_size_bytes: u64, free_vram: u64) -> i32 {
    if free_vram == 0 {
        return 0;
    }
    
    let overhead_factor = 1.25;
    let kv_cache_per_layer = 8 * 1024 * 1024;
    let typical_layers = 32;
    
    let model_with_overhead = (model_size_bytes as f64 * overhead_factor) as u64;
    let available_for_layers = free_vram.saturating_sub(kv_cache_per_layer * typical_layers / 4);
    
    if available_for_layers == 0 {
        return 0;
    }
    
    if available_for_layers > model_with_overhead {
        return 99;
    }
    
    let ratio = available_for_layers as f64 / model_size_bytes as f64;
    let estimated_layers = (typical_layers as f64 * ratio) as i32;
    
    estimated_layers.clamp(0, 99)
}

#[allow(dead_code)]
pub fn estimate_gpu_layers_advanced(
    model_size_bytes: u64, 
    free_vram: u64,
    context_length: u32,
    hidden_size: usize,
    num_attention_heads: usize,
) -> i32 {
    if free_vram == 0 {
        return 0;
    }

    let bytes_per_float = 2;
    let kv_cache_size = 2 * num_attention_heads * context_length as usize * hidden_size * bytes_per_float;
    let overhead = 100 * 1024 * 1024;
    
    let available_for_model = free_vram.saturating_sub(kv_cache_size as u64 + overhead);
    
    if available_for_model == 0 {
        return 0;
    }
    
    if available_for_model > model_size_bytes {
        return 99;
    }
    
    let ratio = available_for_model as f64 / model_size_bytes as f64;
    let typical_layers = 32;
    
    ((typical_layers as f64 * ratio) as i32).clamp(0, 99)
}

#[allow(dead_code)]
pub fn get_optimal_gpu_config(model_size: u64) -> GpuConfig {
    let gpus = discover_gpus();
    
    if gpus.is_empty() {
        return GpuConfig::cpu_only();
    }
    
    let nvidia_gpus: Vec<_> = gpus.iter().filter(|g| g.vendor == "nvidia").collect();
    
    if !nvidia_gpus.is_empty() {
        let best_gpu = nvidia_gpus.iter()
            .max_by_key(|g| g.free_vram)
            .unwrap();
        
        let gpu_layers = estimate_gpu_layers(model_size, best_gpu.free_vram);
        
        return GpuConfig {
            use_gpu: gpu_layers > 0,
            gpu_layers,
            main_gpu: 0,
            tensor_split: None,
            split_mode: SplitMode::Layer,
            gpu_name: Some(best_gpu.name.clone()),
            estimated_vram_usage: if gpu_layers > 0 {
                Some((model_size as f64 * gpu_layers as f64 / 99.0 * 1.2) as u64)
            } else {
                None
            },
        };
    }
    
    let amd_gpus: Vec<_> = gpus.iter().filter(|g| g.vendor == "amd").collect();
    if !amd_gpus.is_empty() {
        let best_gpu = amd_gpus.iter()
            .max_by_key(|g| g.free_vram)
            .unwrap();
        
        let gpu_layers = estimate_gpu_layers(model_size, best_gpu.free_vram);
        
        return GpuConfig {
            use_gpu: gpu_layers > 0,
            gpu_layers,
            main_gpu: 0,
            tensor_split: None,
            split_mode: SplitMode::Layer,
            gpu_name: Some(best_gpu.name.clone()),
            estimated_vram_usage: if gpu_layers > 0 {
                Some((model_size as f64 * gpu_layers as f64 / 99.0 * 1.2) as u64)
            } else {
                None
            },
        };
    }
    
    GpuConfig::cpu_only()
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GpuConfig {
    pub use_gpu: bool,
    pub gpu_layers: i32,
    pub main_gpu: i32,
    pub tensor_split: Option<Vec<f32>>,
    pub split_mode: SplitMode,
    pub gpu_name: Option<String>,
    pub estimated_vram_usage: Option<u64>,
}

#[allow(dead_code)]
impl GpuConfig {
    pub fn cpu_only() -> Self {
        Self {
            use_gpu: false,
            gpu_layers: 0,
            main_gpu: 0,
            tensor_split: None,
            split_mode: SplitMode::None,
            gpu_name: None,
            estimated_vram_usage: None,
        }
    }
    
    pub fn full_gpu(vram: u64) -> Self {
        Self {
            use_gpu: true,
            gpu_layers: 99,
            main_gpu: 0,
            tensor_split: None,
            split_mode: SplitMode::Layer,
            gpu_name: None,
            estimated_vram_usage: Some(vram),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum SplitMode {
    None,
    Layer,
    Row,
}

#[allow(dead_code)]
pub fn print_gpu_info() {
    let gpus = discover_gpus();
    
    if gpus.is_empty() {
        println!("No GPUs detected");
        return;
    }
    
    println!("Detected {} GPU(s):", gpus.len());
    println!("{:-<80}", "");
    
    for (i, gpu) in gpus.iter().enumerate() {
        println!("GPU {}: {}", i, gpu.name);
        println!("  Vendor: {}", gpu.vendor.to_uppercase());
        println!("  VRAM: {} total, {} free", 
            format_bytes(gpu.total_vram),
            format_bytes(gpu.free_vram)
        );
        if let Some(ref cc) = gpu.compute_capability {
            println!("  Compute Capability: {}", cc);
        }
        if let Some(ref driver) = gpu.driver_version {
            println!("  Driver: {}", driver);
        }
        if let Some(ref cuda) = gpu.cuda_version {
            println!("  CUDA: {}", cuda);
        }
        println!();
    }
}

#[allow(dead_code)]
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.0} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[allow(dead_code)]
pub fn has_cuda_support() -> bool {
    discover_gpus().iter().any(|g| g.vendor == "nvidia")
}

#[allow(dead_code)]
pub fn has_rocm_support() -> bool {
    discover_gpus().iter().any(|g| g.vendor == "amd")
}

#[allow(dead_code)]
pub fn get_best_gpu() -> Option<GpuInfo> {
    let gpus = discover_gpus();
    gpus.into_iter().max_by_key(|g| g.free_vram)
}
