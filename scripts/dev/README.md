# Developer diagnostics

Shell scripts for checking the training machinery on this workstation. Both use the GPU
and activate the `mujoco` conda env themselves. Run them from anywhere: each finds the
repository root from its own location. The optional first argument is the folder for
logs and runs. It defaults to a folder under `outputs/debug/`, which git ignores.

| Script | What it checks | Run |
|---|---|---|
| `stall_test.sh` | The stall watchdog and resuming. It starts a real SuperMimic run (256 envs), freezes the training process with `SIGSTOP` after iteration 200, and checks that the supervisor ends the run with exit code 75. It then runs `--continue-run` for about 20 s. It exits 0 only if the exit code was 75, `ckpt_latest.pt` exists, and the continued run resumed from iteration 100 or later. | `scripts/dev/stall_test.sh [work_dir]` (default `outputs/debug/stall_test`) |
| `run_stages.sh` | The local smoke pipeline (`SMOKE=1 hpc/submit_pipeline.sh --local`), one stage at a time. Each stage gets up to 3 attempts with a 50-minute limit each, because this machine has random one-off process failures. Each attempt's log is kept as `stage_<name>_<attempt>.log`. It stops at the first stage that fails 3 times. | `scripts/dev/run_stages.sh [log_dir]` (default `outputs/debug/run_stages`); set `STAGES="ppo evaluate"` to run only some stages |

Neither script should run while another job is using the GPU.
