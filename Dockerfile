# Builder Stage: Use Ubuntu 22.04 for better glibc compatibility with Alpine gcompat
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    clang \
    libclang-dev \
    git \
    pkg-config \
    libssl-dev \
    file \
    binutils \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /usr/src/ollama

# Install dependencies first for better caching
COPY Cargo.toml Cargo.lock ./

# Create a dummy source to pre-build dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    touch src/lib.rs && \
    cargo build --release && \
    rm -rf src

# Now copy the real source and build
COPY . .
ENV CUDA_COMPUTE_CAP=89
RUN cargo build --release --features cuda

# Runtime Stage: Official CUDA Runtime for maximum compatibility
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libvulkan1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy our Rust binary
COPY --from=builder /usr/src/ollama/target/release/ollama /usr/bin/ollama

# Environment variables for NVIDIA Container Toolkit
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_MODELS=/root/.ollama/models

EXPOSE 11434

ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
