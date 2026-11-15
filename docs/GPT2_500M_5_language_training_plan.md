# GPT-2 from scratch: 500M-token, five-language training plan

## 1. Goal and scope

Train a shared multilingual GPT-2-style model from random initialization on English, Vietnamese, Indonesian, Japanese, and Mandarin Chinese. This is the dense baseline for later experiments that divide computation into shared and language-specific parts.

The total training budget remains approximately **500M tokens**, divided equally: **100M tokens per language**. Validation and test text are additional and never used for model or tokenizer training.

Use `id` for Indonesian in code and metadata. The requested `in` refers to Indonesian; `id` is its current language code. The Chinese source below represents Mandarin written in Han characters, rather than every Chinese language variety.

This document is a proposed experiment plan. Hyperparameters and runtime examples are starting choices, not measured results or guarantees. A 500M-token run is a useful pilot; it does not establish that the model is fully trained or will perform well on demanding downstream tasks.

## 2. Fixed experimental configuration

| Component | Setting |
|---|---|
| Language codes | `en`, `vi`, `id`, `ja`, `zh` |
| Language sampling | Equal input-token exposure: 20% each |
| Transformer layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| FFN intermediate size | 3,072 |
| Activation | GPT-2 `gelu_new` |
| Context length | 1,024 tokens |
| Vocabulary | 64,000 including special tokens |
| Embeddings | Shared input embedding and output projection weights |
| Parameter count | Approximately 135M; print the actual count at initialization |
| Initialization | Random; no pretrained weights |
| Training | All parameters trainable |
| Hardware assumption | One NVIDIA A100 40GB |
| Precision | BF16 |
| Data-selection seed | 42 |
| Model initialization seed | 1234 |

The architecture follows GPT-2 Small, but its larger vocabulary changes its parameter count relative to the original approximately 124M-parameter model. Keep vocabulary size fixed across subsequent model comparisons.

## 3. Data sources and collection

Use released web corpora instead of starting with a new crawler.

| Language | Local code | Hugging Face dataset | Configuration |
|---|---|---|---|
| English | `en` | `HuggingFaceFW/fineweb` | `sample-10BT` |
| Vietnamese | `vi` | `HuggingFaceFW/fineweb-2` | `vie_Latn` |
| Indonesian | `id` | `HuggingFaceFW/fineweb-2` | `ind_Latn` |
| Japanese | `ja` | `HuggingFaceFW/fineweb-2` | `jpn_Jpan` |
| Mandarin Chinese | `zh` | `HuggingFaceFW/fineweb-2` | `cmn_Hani` |

### Collection procedure

1. Pin and record a dataset revision for each source.
2. Stream or download selected shards rather than the complete dataset.
3. Sample across multiple source files and crawl snapshots where available. Avoid selecting only the first documents in the source order.
4. Start with approximately 1GB of uncompressed text per language as an initial candidate pool.
5. Preserve document IDs, URLs, source filenames, crawl metadata when available, and dataset revision.
6. After tokenizer training, count tokens and expand the pool if a language cannot supply the training and held-out budgets.

The 1GB target is a collection convenience. It is not a language-independent conversion to tokens. Buffered streaming shuffling is approximate; record the shard selection, shuffle seed, and buffer size.

Suggested document schema:

```json
{
  "doc_id": "source-specific-stable-id",
  "language": "vi",
  "text": "Document text",
  "url": "source-url",
  "source_file": "source-shard",
  "dataset_revision": "pinned-revision"
}
```

Keep the source dataset metadata and attribution information alongside the corpus. Record that English and the other languages come from different dataset pipelines; this is a possible source of differences in corpus composition.

## 4. Cleaning and quality inspection

Apply conservative additional cleaning because FineWeb and FineWeb2 already include preprocessing.

1. Normalize text to Unicode NFC and normalize line endings.
2. Remove empty documents, invalid text, and obvious extraction failures.
3. Detect documents dominated by duplicated lines, repeated character sequences, URLs, or boilerplate.
4. Review language identification and mixed-language documents.
5. Remove exact duplicates using a hash of normalized text.
6. Detect near duplicates where feasible and retain duplicate-group IDs for split assignment.

Preserve Vietnamese diacritics, Japanese scripts, Chinese characters, punctuation, and ordinary formatting. Do not lowercase everything or remove accents. Do not impose English-style whitespace word-count thresholds on Japanese and Chinese. Treat short-line and script-ratio filters cautiously, and inspect their rejected examples.

For each language, manually inspect 50–100 kept documents and a smaller sample of rejected documents. Set cleaning thresholds from these inspections and record them before final selection.

Produce a cleaning summary with collected/kept document counts, rejection reasons, duplicate rate, and retained UTF-8 bytes.

## 5. Train, validation, and test splits

Split documents before training the tokenizer or packing tokens.

1. Group detected duplicates and near duplicates.
2. Assign complete groups to splits using a deterministic hash and split seed.
3. Initially reserve approximately 96% for training candidates, 2% for validation, and 2% for test.
4. Train the tokenizer only on training candidates.
5. Tokenize the reserved pools and create the final held-out subsets.

| Split | Target per language | Total across five languages |
|---|---:|---:|
| Training | Approximately 100M tokens | Approximately 500M tokens |
| Validation | Approximately 1M tokens | Approximately 5M tokens |
| Test | Approximately 1M tokens | Approximately 5M tokens |

A convenient held-out size is 1,024 blocks per language, or 1,048,576 input tokens. Use a fixed subset of about 100 blocks per language for quick validation.

Check document IDs and duplicate groups for overlap between splits. Reserve test data for final evaluation; do not repeatedly use it to choose settings. Apply the same split assignment to documents collected later.

## 6. Train the shared tokenizer

### Configuration

| Setting | Choice |
|---|---|
| Algorithm | Byte-level BPE |
| Vocabulary | 64,000 including special tokens |
| Normalizer | NFC |
| Pre-tokenizer | ByteLevel |
| Decoder | ByteLevel |
| Initial alphabet | All 256 byte values |
| Special tokens | `<pad>`, `<bos>`, `<eos>` |
| Tokenizer-training sample | Approximately 100MB of UTF-8 text per language |

Sample tokenizer text from training candidates only, using a fixed seed. Equal UTF-8 bytes are a simple initial rule, but they do not represent equal linguistic content. Vocabulary entries may be shared across languages; do not force a fixed vocabulary allocation per language for this baseline.

### Validation before freezing

- Check encode/decode round trips against normalized input text.
- Inspect sentence, name, number, and punctuation tokenizations in all five languages.
- Calculate tokens per Unicode character and per UTF-8 byte on fixed samples.
- Measure tokenized document-length distributions.
- Inspect fragmentation in Vietnamese, Japanese, and Chinese.
- Verify special-token IDs and prevent literal corpus strings from accidentally becoming control tokens during encoding.
- Confirm the final vocabulary size and save all tokenizer configuration files.

Once these checks pass, freeze the tokenizer. Do not retrain it separately for each architecture. Count final training tokens using this tokenizer, rather than the source dataset's reported token counts.

## 7. Tokenization and packing

For each language independently:

1. Shuffle training documents deterministically.
2. Encode documents without automatic BOS/EOS insertion.
3. Append one EOS token to each document.
4. Concatenate the token streams within that language.
5. Divide the stream into 1,024-token blocks.
6. Carry leftover tokens between preprocessing batches; drop only the final insufficient remainder after selection.
7. Store each block with its language code and a stable block identifier.

Keep blocks monolingual. This supplies clear routing labels for the later partitioned models. Mixed-language batches are acceptable when routing can operate per example.

Use ordinary causal attention across packed documents for this baseline. An EOS separator marks a boundary but does not block attention to the preceding document. Keep this behavior identical across model comparisons, or explicitly implement document-isolated attention in all conditions.

### Practical budget using complete 1,024-token blocks

Select 97,656 full blocks per language. This gives 99,999,744 tokens per language, just 256 tokens below the 100M target. Keep the final shorter optimizer batch so that no selected blocks are dropped:

| Quantity | Value |
|---|---:|
| Blocks per language | 97,656 |
| Tokens per language | 99,999,744 |
| All-language blocks | 488,280 |
| All-language input tokens | 499,998,720 |
| Difference from 500M | 1,280 tokens, approximately 0.000256% |

Each language receives exactly 20% of the training tokens. The 256-token shortfall per language is due only to rounding down to complete blocks; it is not an additional reduction in the intended 100M-token budget. Refer to the experiment as approximately 500M tokens, and report the exact count in results.

Store token IDs as `uint16` if the final vocabulary fits within 65,536 values. Convert batches to the dtype required by the model, normally `torch.long`, when loading them.

## 8. Model initialization

After loading the newly trained tokenizer:

```python
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=3072,
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    tie_word_embeddings=True,
    use_cache=False,
)

model = GPT2LMHeadModel(config)
print(sum(parameter.numel() for parameter in model.parameters()))
```

Set the model seed before construction. Do not load GPT-2 pretrained weights. The snippet assumes `tokenizer` is already loaded and is an initialization example, not a complete training script.

## 9. Training objective and batching

Train on next-token prediction with all parameters trainable. With Hugging Face `GPT2LMHeadModel`, set unshifted labels equal to input IDs; the model handles the target shift internally. Do not shift labels a second time.

Full training blocks need no padding. When padding is used for other inputs, set its labels to `-100` and supply the appropriate attention mask. A separate PAD token avoids accidentally masking real EOS targets.

Count the budget as input tokens processed. A 1,024-token block normally contributes 1,023 next-token targets in this implementation. Use the same counting convention in every experiment.

| Batch setting | Value |
|---|---:|
| GPUs | 1 |
| Microbatch | 8 sequences |
| Gradient accumulation | 16 microbatches |
| Effective batch | 128 sequences; final batch 88 sequences |
| Input tokens per full optimizer step | 131,072 |
| Total optimizer steps | 3,815 |
| Total microbatches | 61,035 |
| Passes over selected training blocks | 1 |

There are 3,814 full optimizer steps plus one final step with 88 sequences (90,112 input tokens). With microbatch 8, the final step accumulates 11 microbatches instead of 16. Set `drop_last=False` and verify the framework actually consumes all final blocks.

Shuffle the complete fixed list of selected blocks and consume it once. This guarantees equal final exposure without requiring every batch to contain an exactly equal language mixture. Avoid replacement sampling for this run.

If memory is insufficient, use microbatch 4 and accumulation 32. Keep the effective batch unchanged. Start without gradient checkpointing; enable it only if needed, since it trades extra computation for memory.

## 10. Optimizer and learning-rate settings

| Hyperparameter | Proposed value |
|---|---|
| Optimizer | AdamW |
| Peak learning rate | `3e-4` |
| Betas | `(0.9, 0.95)` |
| Epsilon | `1e-8` |
| Weight decay | `0.1` |
| Weight-decay exclusions | Biases and LayerNorm parameters |
| Gradient clipping | `1.0` |
| Precision | BF16 autocast |
| Warmup | First 77 optimizer steps |
| Schedule after warmup | Cosine decay |
| Final learning rate | `3e-5` |
| Training cache | Disabled |

Divide the microbatch loss by the actual accumulation count when manually accumulating gradients: 16 for full optimizer steps and 11 for the final step with microbatch 8. If microbatch size changes or the last microbatch is smaller, normalize the final accumulation by its valid-target count instead. Apply clipping, the optimizer update, and scheduler update once per optimizer step. If using a framework, let it handle accumulation and do not divide the loss again externally.

A cosine schedule must explicitly end at 10% of peak learning rate to reach `3e-5`; some default cosine schedules decay to zero. Verify the actual schedule during the smoke test.

## 11. Debugging and launch checks

Perform these checks using disposable model instances before the final run:

| Check | Procedure | Expected outcome |
|---|---|---|
| Tiny-batch overfitting | Repeatedly train on a few fixed blocks | Loss decreases substantially |
| Smoke test | Train for 100–200 optimizer steps | Finite loss, correct logs and evaluation |
| Resume test | Save, restart, and continue | Optimizer, scheduler, RNG, and data position restore |
| Data accounting | Count selected and consumed blocks by language | Equal final exposure with no unintended repeats |

Measure GPU memory, gradient norms, actual learning rate, and steady-state throughput. Inspect decoded examples from loaded training batches to catch serialization or token-ID errors.

After debugging, restart the final model from seed 1234 with a fresh optimizer and scheduler. Do not include debugging updates in the baseline budget.

## 12. Monitoring, evaluation, and checkpoints

| Operation | Frequency |
|---|---|
| Training statistics | Every 10–20 optimizer steps |
| Quick validation: about 100 blocks/language | Every 250 steps |
| Full validation: 1,024 blocks/language | Every 1,000 steps and at the end |
| Resumable checkpoint | Every 500 steps and at the end |
| Fixed-prompt generation | At initialization and selected milestones |
| Test evaluation | After model selection |

Use loss-only validation, aggregating loss sums and valid-target counts rather than storing full vocabulary logits. Run evaluation with dropout disabled.

Report per-language cross-entropy and perplexity. Compute each language's mean loss over its valid prediction targets, then take the equal-weight average of the five language losses for the macro loss. Perplexity for a language is `exp(mean_loss)`.

Compare model variants within each language using the same tokenizer and held-out text. Cross-language raw perplexity rankings are affected by tokenization and should not be interpreted directly as language ability.

| Optimizer step | Cumulative input tokens |
|---|---:|
| 250 | 32,768,000 |
| 1,000 | 131,072,000 |
| 2,000 | 262,144,000 |
| 3,000 | 393,216,000 |
| 3,814 | 499,908,608 |
| 3,815 | 499,998,720 |

Retain the latest two or three resumable checkpoints, the final checkpoint, and selected milestone models as storage permits. Each resumable checkpoint should include model, optimizer, scheduler, RNG states, consumed-block position, and configuration. Checkpoints should represent completed optimizer steps.

For later architecture comparisons, use the final equal-token-budget checkpoint as the primary result. A validation-selected checkpoint can be an additional result if the selection rule is identical across models.

## 13. Runtime and storage planning

Measure throughput on the actual A100 job after warmup. Include the forward pass, backward pass, accumulation, and optimizer updates in the timing window. Use at least 100 steady-state microbatches, and time validation and checkpoint saving separately.

`training_hours = 499,998,720 / (measured_input_tokens_per_second * 3600)`

| Measured throughput | Training time excluding evaluation/checkpointing |
|---:|---:|
| 5,000 tokens/s | 27.8 hours |
| 10,000 tokens/s | 13.9 hours |
| 20,000 tokens/s | 6.9 hours |
| 30,000 tokens/s | 4.6 hours |

These are conditional arithmetic examples, not measured A100 estimates. Add observed evaluation/checkpoint time when requesting the final job allocation. Preprocess data before allocating the training GPU whenever possible.

The packed training token array alone is about 1.0GB using `uint16`, or about 2.0GB using `int32`. Allow additional space for source text, metadata, held-out data, tokenizer files, logs, and resumable checkpoints. Check actual checkpoint size during the smoke test before deciding retention.

## 14. Suggested implementation files

| File | Responsibility |
|---|---|
| `configs/data.yaml` | Sources, language mapping, revisions, filters, split seed |
| `configs/train.yaml` | Model, optimizer, batch settings, token budget |
| `prepare_data.py` | Collect, normalize, deduplicate, split, and report |
| `train_tokenizer.py` | Train and save the shared tokenizer |
| `inspect_tokenizer.py` | Round trips, fragmentation, and length statistics |
| `pack_tokens.py` | Create monolingual blocks and selection manifests |
| `train.py` | Initialize, train, evaluate, checkpoint, and resume |
| `evaluate.py` | Per-language held-out loss and fixed-prompt generation |
| `environment-lock.txt` | Exact tested package versions |
| `README.md` | Commands and reproduction instructions |

This is a proposed project layout. The scripts themselves are a separate implementation task.

## 15. Work order and completion criteria

1. **Collect candidate pools:** all five sources resolved and revisions recorded.
2. **Clean and split:** samples inspected; duplicate groups do not cross splits.
3. **Train tokenizer:** all five languages inspected; tokenizer frozen.
4. **Pack data:** exactly 97,656 training blocks per language selected.
5. **Debug:** tiny-batch, smoke, and resume checks pass.
6. **Benchmark:** measured throughput and checkpoint sizes recorded.
7. **Train:** complete 3,815 optimizer steps from a fresh initialization.
8. **Evaluate:** final per-language validation/test results and generation samples saved.

### Final deliverables checklist

- [ ] Dataset revisions and source selection recorded.
- [ ] Cleaning summary and split manifests saved.
- [ ] Shared tokenizer and its settings saved.
- [ ] Exactly equal training-token exposure verified for all five languages.
- [ ] Model configuration and actual parameter count saved.
- [ ] Tested software versions and seeds recorded.
- [ ] Final weights and a resumable checkpoint saved.
- [ ] Per-language learning curves and held-out results saved.
- [ ] Actual tokens processed, GPU time, memory, and throughput reported.
- [ ] Final comparison protocol freezes data, tokenizer, token budget, and evaluation.

For the partitioned model, additionally report total parameters, active parameters per language, and measured compute. The shared baseline does not itself demonstrate language-specific knowledge localization.

## 16. Reference documentation

- [Hugging Face GPT-2 documentation](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [Hugging Face Trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Hugging Face Tokenizers quicktour](https://huggingface.co/docs/tokenizers/quicktour)
- [FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [FineWeb2 dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)

Dataset configuration names and API behavior should be checked against the pinned revisions and tested package versions used for implementation.
