#  Multi‑GPU / Multi‑Node Distributed Training

This repository implements a distributed training pipeline for a sequence-to-sequence Transformer with multi-head attention. 

**Distributed training methodologies (overview)**
This project currently uses DDP. Below is a concise description of several paradigms you may consider when scaling beyond single-node multi-GPU training.

- **Data Parallelism (DDP)**
   - Description: Each process holds a full copy of the model and processes a shard of the input data. Gradients are synchronized across processes (all-reduce) after backward().
   - Pros: Simple to reason about; works with standard PyTorch optimizers; excellent inter-op support and mature tooling.
   - Cons: Memory-limited by a single device (model must fit on one GPU); communication overhead for gradient synchronization as model size grows.


- **Fully Sharded Data Parallel (FSDP)**
   - Description: Partitions model states (parameters, gradients, optimizer states) across ranks to reduce per-GPU memory footprint. Sharded pieces are moved and gathered on-demand during forward/backward.
   - Pros: Much lower memory usage per GPU; enables training larger models without model-parallel code.
   - Cons: More complex runtime and tuning (sharding strategies, CPU vs GPU offload); some collective overheads can affect throughput.
   - Integration notes: Replace DDP wrapper with `torch.distributed.fsdp.FullyShardedDataParallel` and adjust checkpointing to aggregate/persist sharded state. Be careful with custom layers that assume full-parameter access.

- **Tensor (model) Parallelism**
   - Description: Splits model tensors (e.g., weight matrices or attention heads) across devices so that layers are computed in parallel across ranks. Communication implements partial-reduce and broadcast patterns inside layers.
   - Pros: Enables very large models where individual layers exceed a single GPU's memory; scales model size horizontally.
   - Cons: Requires library support (e.g., Megatron-LM style utilities or frameworks like DeepSpeed tensor parallel), added complexity to layer implementations, and careful pipeline scheduling.
   - Consider using a framework (Megatron, DeepSpeed TP) rather than hand-rolling primitives.

- **Expert/ Mixture-of-Experts (MoE) Parallelism**
   - Description: Model contains multiple 'expert' sub-networks. A gating network routes tokens to a small subset of experts per token, so each token is processed by few experts while the overall capacity grows.
   - Pros: Increased model capacity with sublinear compute cost per token; efficient for sparse compute patterns.
   - Cons: Complex routing, load balancing, and communication (all-to-all) between devices; requires careful implementation to avoid stragglers.
   - Libraries like DeepSpeed and fairscale provide MoE building blocks and routing utilities.


**High-level Architecture**
- **Model**: Standard encoder-decoder Transformer architecture (scaled dot-product multi-head attention, position-wise feed-forward networks, residual connections and layer normalization). The model exposes `encode`, `decode` and a final `project` projection layer; training uses the combined forward wrapper for efficiency.
- **Multi-head attention**: Implemented per the original Transformer paper; attention outputs are combined and projected back to model dimension `d_model`.

**Distributed Training Design**
- **Process model**: Uses PyTorch Distributed Data Parallel (`torch.distributed.init_process_group`, `DistributedDataParallel`) with one process per GPU. Devices are selected using `local_rank` and `torch.cuda.set_device`.
- **Data sharding**: Training `DataLoader` uses `DistributedSampler` to ensure each process receives a distinct partition of the dataset per epoch.
- **Checkpointing**: Checkpoints save `epoch`, `model_state_dict` (taken from `model.module` when wrapped in DDP), optimizer state, and `global_step`. Only the main/global rank writes checkpoints to avoid races.

**Data pipeline**
- **Source**: Uses Hugging Face `datasets` (`opus_books`) as the raw text source and a small WordLevel tokenizer built with `tokenizers` when no tokenizer file exists.
- **Tokenization**: `get_or_build_tokenizer` constructs or loads a `tokenizer` per language; tokenizers are persisted to JSON files for inference reproducibility.
- **Batching & Masks**: `BilingualDataset` returns `encoder_input`, `decoder_input`, `encoder_mask`, `decoder_mask`, and `label` tensors. `causal_mask` is used to enforce autoregressive decoding during training/validation.

**Training loop & optimization**
- **Loss**: Cross-entropy with `ignore_index` set to the PAD token and `label_smoothing=0.1`.
- **Optimizer**: Adam with small `eps` for numerical stability.
- **Grad/update flow**: Standard backward(), optimizer.step(), and zeroing gradients. The code expects to run under DDP, so parameter synchronization is handled by PyTorch.

**Validation & Metrics**
- **Greedy decoding**: Validation uses a `greedy_decode` routine that reuses encoder outputs and steps the decoder until `EOS` or `max_len`.
- **Metrics**: Uses `torchmetrics` for Character Error Rate (CER), Word Error Rate (WER), and BLEU. Metrics are computed on the rank-0 process and printed by default.

**Configuration & Extensibility**
- **`ModelConfig`**: Central configuration dataclass (batch size, learning rate, sequence lengths, model dims, file locations). Command-line args overlay defaults at startup.
- **Checkpoint resume**: The checkpoint-loading code populates `initial_epoch` and `global_step` from saved state to resume training seamlessly.
- **Instrumenting**: The code centralizes rank-0 logging and metric computation to avoid duplicated outputs from all processes. Replacing or adding an experiment backend (e.g., MLFlow) can be done by plugging an optional logger on rank 0.


