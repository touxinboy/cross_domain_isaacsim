# Vendored source changes

## 2026-10-30: remove embedded CI credential

`OmniNav/train_code/transformers-main/src/transformers/testing_utils.py`
contained an upstream Hugging Face staging CI token. The test utility now reads
`TRANSFORMERS_TEST_HUB_TOKEN` from the environment, defaulting to an empty string.
Authenticated upstream hub tests require this environment variable to be supplied
locally. This change does not alter model inference or the waypoint head.

The source URLs, pinned commits and selected original file hashes remain recorded
in `../configs/navigation_sources.json`.
