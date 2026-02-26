# 🦙 Ollama Rust

Uma implementação em Rust de alta performance do Ollama, projetada para rodar modelos de linguagem (LLMs) localmente com eficiência máxima. Este projeto utiliza o ecossistema [Candle](https://github.com/huggingface/candle) da Hugging Face para inferência e foi otimizado para ambientes conteinerizados.

---

## 🚀 Visão Geral

O **Ollama Rust** oferece uma alternativa leve e extremamente rápida para execução de LLMs. Ao contrário de implementações baseadas em frameworks mais pesados, o Ollama Rust foca em baixa latência, gestão eficiente de memória e suporte nativo a aceleração por hardware (CUDA/Vulkan).

### ✨ Principais Funcionalidades

- **Inferência Nativa**: Motor de inferência puramente em Rust.
- **Aceleração GPU**: Suporte robusto para NVIDIA CUDA.
- **API Compatível**: Interface REST compatível com o Ollama original.
- **Suporte a Modelos**: Compatibilidade com formatos GGUF e Safetensors.
- **Eficiência de Recursos**: Menor pegada de memória e CPU em comparação com binários Go/C++.
- **WASM Ready**: Capacidade experimental de rodar em navegadores via WebAssembly.

---

## 🛠️ Arquitetura do Projeto

O projeto é estruturado de forma modular para facilitar a manutenção e escalabilidade:

- **`src/core/`**: O coração do sistema, lidando com o carregamento de pesos, gerenciamento de tokens e execução do modelo.
- **`src/api/`**: Servidor HTTP (Axum) que implementa os endpoints do Ollama.
- **`src/runner/`**: Orquestrador que gerencia o ciclo de vida da execução dos modelos.
- **`src/tokenizer/`**: Implementações de tokenização rápidas e seguras.
- **`src/infra/`**: Camada de gerenciamento de hardware (detecção de GPUs, monitoramento de VRAM).

---

## 📦 Instalação e Uso

### Pré-requisitos

- Rust 1.75 ou superior.
- (Opcional) NVIDIA CUDA Toolkit 12.x para aceleração por GPU.
- CMake e Compiladores C/C++ (para dependências nativas).

### Compilação

Para compilar a versão otimizada com suporte a CUDA:

```bash
cargo build --release --features cuda
```

Para uma versão CPU-only (mais lenta, mas universal):

```bash
cargo build --release
```

---

## 🐳 Docker

O projeto foi desenhado para ser executado em containers. O `Dockerfile` utiliza uma abordagem multi-stage para gerar imagens mínimas e seguras.

### Rodando com Docker Compose

No diretório raiz das integrações:

```bash
docker compose up -d ollama
```

### Configurações de GPU no Docker

Certifique-se de ter o `nvidia-container-toolkit` instalado no host. O `docker-compose.yml` já está configurado para expor todas as GPUs disponíveis para o container.

---

## ⚙️ Variáveis de Ambiente

O **Ollama Rust** pode ser configurado através de variáveis de ambiente:

| Variável | Descrição | Padrão |
| :--- | :--- | :--- |
| `OLLAMA_HOST` | Host e porta para o servidor API | `0.0.0.0:11434` |
| `OLLAMA_MODELS` | Diretório para armazenamento dos modelos | `/home/ollama/.ollama/models` |
| `OLLAMA_KEEP_ALIVE` | Tempo que o modelo permanece em VRAM | `30m` |
| `OLLAMA_NUM_PARALLEL` | Número de requisições paralelas | `1` |
| `CUDA_VISIBLE_DEVICES` | IDs das GPUs visíveis para o processo | `all` |

---

## 📝 Roadmap e Contribuição

Atualmente, o projeto foca na estabilidade da API e suporte a novos arquiteturas de modelos. Contribuições são bem-vindas!

1. Faça um Fork do projeto.
2. Crie uma Branch para sua feature (`git checkout -b feature/minha-melhoria`).
3. Faça o Commit de suas alterações (`git commit -m 'Adiciona funcionalidade X'`).
4. Faça o Push para a Branch (`git push origin feature/minha-melhoria`).
5. Abra um Pull Request.

---

## 📜 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

---

**Desenvolvido com ❤️ pela equipe do Integracoes.**
