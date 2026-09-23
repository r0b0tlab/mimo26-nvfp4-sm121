FROM lmsysorg/sglang:nightly-dev-cu13-20260922-582389ce
# Environment fix only: the nightly image lacks torchcodec, so sglang silently
# drops the MiMo-V2 multimodal processor at import time and every boot dies
# with "No processor registered for architecture: ['MiMoV2ForCausalLM']".
# No source patches, no upstream code changes.
RUN pip install --no-cache-dir torchcodec
