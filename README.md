# safety-probes-under-shift

Do linear probes on an LLM's hidden states still detect harmful requests when the input shifts?

**Setup**
- Model: Qwen2.5-0.5B-Instruct
- Feature: last-token hidden state after the chat template
- Probe: logistic regression (harmful = 1, benign = 0)

**Data** (`safety_probe_prompts.csv`, 96 prompts, 48/48 balanced)
- `train` (48) and `test_iid` (16): direct prompts
- `test_shift` (32): paraphrases, typos, and Vietnamese translations
- Includes hard negatives: benign prompts that use sensitive words (e.g. asking why explosives are dangerous)

**Status:** work in progress. Hidden-state extraction works; probe training and a per-layer sweep are next.
Notes: [research_log.md](research_log.md) · blog: tuanairesearch.com

**Limitations:** very small dataset; results will be noisy.