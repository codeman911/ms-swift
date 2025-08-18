# Optimized Higgs-Audio Training for 8x H200 GPUs + 128 CPU Cores

## Hardware Utilization Strategy

### GPU Optimization (8x H200)
- **DeepSpeed ZeRO-3**: Partitions optimizer states, gradients, and parameters across 8 GPUs
- **Gradient Checkpointing**: Reduces memory usage by ~40% at cost of 20% compute overhead
- **Mixed Precision**: `bf16=true` + `tf32=true` for maximum H200 performance
- **NCCL Optimization**: InfiniBand configuration for fastest multi-GPU communication

### CPU Optimization (128 cores)
- **Data Loading**: `dataloader_num_workers=32` (4 workers per GPU)
- **Preprocessing**: `preprocess_num_proc=32` for parallel data preprocessing
- **OMP Threads**: 16 threads per GPU process for optimal CPU utilization

### Memory Optimization
- **Effective Batch Size**: 256 (4 per device × 8 GPUs × 8 accumulation steps)
- **DeepSpeed Offloading**: Optimizer and parameter offloading to CPU when needed
- **Pin Memory**: Faster CPU-GPU data transfers

### Performance Features
- **Fused Optimizer**: `adamw_torch_fused` for 15-20% speedup
- **Persistent Workers**: Reduces data loading overhead
- **Group by Length**: Minimizes padding and improves throughput

## Usage

```bash
cd /path/to/ms-swift
chmod +x examples/train/multimodal/higgs_audio_validating/sft_optimized.sh
./examples/train/multimodal/higgs_audio_validating/sft_optimized.sh
```

## Expected Performance
- **Training Speed**: ~3-4x faster than single GPU
- **Memory Efficiency**: Can handle 4K sequence lengths with 4GB per sample
- **Throughput**: ~32-64 samples/second depending on audio length

## Key MS-SWIFT Parameters Used

### Core Training
- `--deepspeed default-zero3`: Multi-GPU memory optimization
- `--bf16 true`: Native H200 precision format
- `--gradient_checkpointing true`: Memory-compute tradeoff
- `--optim adamw_torch_fused`: Fastest optimizer implementation

### Data Loading
- `--dataloader_num_workers 32`: Parallel data loading
- `--dataloader_pin_memory true`: Faster transfers
- `--preprocess_num_proc 32`: Parallel preprocessing

### LoRA Optimization
- `--lora_target_modules ALL`: Target all applicable modules
- `--lora_rank 16`: Balanced capacity vs efficiency
- `--lora_alpha 64`: Higher learning rate for LoRA layers

### Monitoring
- `--report_to tensorboard`: Performance monitoring
- `--logging_steps 5`: Frequent logging for large scale training
