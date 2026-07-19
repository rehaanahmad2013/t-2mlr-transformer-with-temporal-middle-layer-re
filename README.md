# T²MLR retrofit reproduction

This project tested two claims from [T²MLR: Transformer with Temporal Middle-Layer Recurrence (arXiv:2607.15178)](https://arxiv.org/abs/2607.15178): that a recurrent SmolLM2-1.7B-Instruct retrofit beats an identically OpenMathReasoning-fine-tuned baseline, and that the recurrent path adds no more than about 8% autoregressive latency and less than 0.1% peak inference memory.

**Verdict: both claims were not attempted.** The paired, revision-pinned implementation was completed, but Kubernetes execution exhausted the 192 reserved GPU-hour limit during runner startup and deterministic dataset construction, before model loading, fine-tuning, evaluation, or profiling. This is a blocked reproduction, not evidence against T²MLR.

The paper reports **GSM8K 35.78→39.88** and **MATH500 12.80→18.00**; this reproduction has **no accuracy, latency, or memory number**. The official pinned SmolLM2 checkpoint has 24 layers rather than the paper's stated 32, so the closest valid path **24→5** replaced **28→5**. The final bounded protocol kept the full 1.7B model but reduced training to 512 OpenMathReasoning examples, Jacobi depth to 8/1, evaluation to 256 GSM8K and 100 MATH500 examples, and profiling to two repeats at 128/512 tokens.

All runs used **OpenResearch Kubernetes** on **NVIDIA RTX PRO 6000 Blackwell Server Edition** GPUs. Peak concurrency was 16 GPUs; the final pair used 4 GPUs each. The six launches reserved **192 GPU-hours total** and finished within the 12-hour paper wall-clock cap. Read the [illustrated reproduction report](reports/t2mlr-reproduction/report.md) for the implementation, evidence, failure analysis, and exact claim verdicts.

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Verdict / outcome | Compute |
|---|---|---|---|---|
| `main` | Publication surface: README and illustrated report | Not run as an experiment (publication surface) | Published provenance and blocked report | No experiment compute |
| [`orx/matched-smollm2-baseline`](https://github.com/rehaanahmad2013/t-2mlr-transformer-with-temporal-middle-layer-re/tree/orx/matched-smollm2-baseline) | Frozen matched control harness | `bash run.sh` | Failed in manifest shell parsing after 5s; no science | Kubernetes, 8× RTX PRO 6000 Blackwell, 6h requested, 48 reserved GPU-hours |
| [`orx/t2mlr-layer-5-to-layer-24-retrofit`](https://github.com/rehaanahmad2013/t-2mlr-transformer-with-temporal-middle-layer-re/tree/orx/t2mlr-layer-5-to-layer-24-retrofit) | Add gated 24→5 recurrence; later retry shell fix | `bash run.sh` | Initial manifest failure after 5s; retry hit lazy-NCCL timeout after 11m04s | Kubernetes, 8× RTX PRO 6000 Blackwell; 6h + 5h requested, 88 reserved GPU-hours |
| [`orx/kubernetes-runner-fixed-baseline`](https://github.com/rehaanahmad2013/t-2mlr-transformer-with-temporal-middle-layer-re/tree/orx/kubernetes-runner-fixed-baseline) | Reparse supervisor script in Kubernetes | `bash run.sh` | Lazy-NCCL barrier timed out after 10m38s | Kubernetes, 8× RTX PRO 6000 Blackwell, 5h requested, 40 reserved GPU-hours |
| [`orx/bounded-4-gpu-paired-baseline`](https://github.com/rehaanahmad2013/t-2mlr-transformer-with-temporal-middle-layer-re/tree/orx/bounded-4-gpu-paired-baseline) | Initialize NCCL early; bounded paired control | `bash run.sh` | `DeadlineExceeded` during dataset construction; no model loaded | Kubernetes, 4× RTX PRO 6000 Blackwell, 2h requested, 8 reserved GPU-hours; elapsed 2h00m |
| [`orx/bounded-4-gpu-t2mlr-retrofit`](https://github.com/rehaanahmad2013/t-2mlr-transformer-with-temporal-middle-layer-re/tree/orx/bounded-4-gpu-t2mlr-retrofit) | Identical bounded child with recurrence enabled | `bash run.sh` | `DeadlineExceeded` during dataset construction; no model loaded | Kubernetes, 4× RTX PRO 6000 Blackwell, 2h requested, 8 reserved GPU-hours; elapsed 2h00m |

The command values above are copied verbatim from `orx exp status`; behavior varies only through committed code/configuration.

---

<details>
<summary>Original repository README</summary>

# t-2mlr-transformer-with-temporal-middle-layer-re

</details>
