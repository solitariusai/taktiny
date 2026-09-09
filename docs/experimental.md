# Experimental APIs

This page separates implemented foundations from unfinished integrations. The
symbols described here may be useful for development, but they are not current
support claims.

## Multimodal Models

`TransformerConditionalGeneration` and several conditional architecture names
exist as scaffolding. The current `experiment` branch has no parity-tested,
supported vision-language or audio-language family.

Gemma3 support currently means `Gemma3ForCausalLM`, the text model, not
`Gemma3ForConditionalGeneration`. A complete multimodal implementation still
needs:

- architecture-specific media towers and projectors
- processor-compatible placeholder and feature merging
- media-aware prefill followed by correct cached text decoding
- complete checkpoint mappings and reference parity tests
- padded-batch and multiple-media generation tests

Do not use registry presence as evidence that these pieces exist.

## vLLM Wrapper

The optional runtime wrapper is isolated from `Maestro` and model classes:

```python
from taktiny.ensembles.vllm import VLLM

runtime = VLLM(
    model,
    tensor_parallel_size=2,
)

output_ids = runtime.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=128,
    temperature=0.7,
)

runtime.sync()
runtime.close()
```

`VLLM` is a runtime wrapper, not an `nn.Module`. `runtime.model` remains the
original trainable Taktiny model. The wrapper owns an inference engine,
delegates generation, tracks policy versions, and synchronizes updated weights.

The bundled `LocalVLLMEngine` currently targets GPU vLLM. It normalizes padded
token batches, preserves the native `generate` result shape, and supports GPU
weight synchronization. Streaming is not implemented by this local offline
engine.

Native Taktiny execution through vLLM TPU is explicitly `NotImplemented`.
vLLM TPU expects its paged KV-cache and `AttentionMetadata` model contract;
substituting a separate Flax NNX model would not execute the supplied Taktiny
module. An external or future TPU integration must implement `VLLMEngine`.

## RL Foundation

`taktiny.trainer` exports:

- `PolicyRuntime`: protocol with `model`, `generate`, and `sync`
- `Rollout`: token output stamped with the policy version that produced it
- `RLBaseTrainer`: generic runtime synchronization and rollout lifecycle

`RLBaseTrainer` inherits `Trainer`, so concrete algorithms still use
`trainer.train()`. For online RL it synchronizes a dirty policy before the next
rollout batch and after the final optimizer update. Stale rollouts are rejected
by policy version.

The base is abstract at the algorithm boundary. A subclass must implement
rollout construction, reward computation, and conversion to an optimizer-ready
batch. Taktiny does not currently ship concrete GRPO, DPO, PPO, or RLOO
trainers.

## Registered Placeholders

The internal architecture repertoire includes development names for DeepSeek,
GPT-OSS, Llama 4, Gemma 4, Qwen MoE/Next/3.5, and diffusion models. Those
registrations make class lookup possible; they do not establish valid
checkpoint mappings, reference parity, generation, or training support.
