fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    
    if target_os == "linux" {
        let lib_cuda = if target_arch == "x86_64" {
            "/usr/lib/x86_64-linux-gnu/libcuda.so"
        } else if target_arch == "aarch64" {
            "/usr/lib/aarch64-linux-gnu/libcuda.so"
        } else {
            ""
        };
        
        if !lib_cuda.is_empty() && std::path::Path::new(lib_cuda).exists() {
            println!("cargo:rustc-link-lib=dylib=cuda");
            println!("cargo:rustc-cfg=feature=\"cuda\"");
        }
        
        if std::path::Path::new("/usr/lib/libvulkan.so").exists() ||
           std::path::Path::new("/usr/lib/x86_64-linux-gnu/libvulkan.so").exists() {
            println!("cargo:rustc-link-lib=dylib=vulkan");
            println!("cargo:rustc-cfg=feature=\"vulkan\"");
        }
    }
}
