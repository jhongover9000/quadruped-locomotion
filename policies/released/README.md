# Released policies

Policies approved for hardware, exported by `scripts/export_policy.py` and tracked with
Git LFS. One folder per release:

    <algo>_<task>_<robot>_v<N>/
        policy.onnx
        config.yaml      # the exact training config
        notes.md         # source run, evaluation results, hardware test outcome
