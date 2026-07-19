# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.14.17"]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    claim_rows = [
        {
            "claim": "Matched recurrent retrofit improves reasoning",
            "paper evidence": "GSM8K 35.78→39.88; MATH500 12.80→18.00",
            "reproduced evidence": "No model load, fine-tune, or accuracy event",
            "verdict": "Not reproduced (measurement not reached)",
        },
        {
            "claim": "Recurrent path adds ≤~8% latency and <0.1% memory",
            "paper evidence": "Reported on the paper's 135M/361M/1B models",
            "reproduced evidence": "No latency or peak-memory event",
            "verdict": "Not reproduced (measurement not reached)",
        },
    ]
    run_rows = [
        {"experiment": "Initial baseline", "GPUs": 8, "requested": "6h", "reserved GPU-h": 48, "elapsed": "5s", "outcome": "Manifest shell parse failure"},
        {"experiment": "Initial T²MLR", "GPUs": 8, "requested": "6h", "reserved GPU-h": 48, "elapsed": "5s", "outcome": "Manifest shell parse failure"},
        {"experiment": "Runner-fixed baseline", "GPUs": 8, "requested": "5h", "reserved GPU-h": 40, "elapsed": "10m38s", "outcome": "Lazy NCCL timeout"},
        {"experiment": "Runner-fixed T²MLR", "GPUs": 8, "requested": "5h", "reserved GPU-h": 40, "elapsed": "11m04s", "outcome": "Lazy NCCL timeout"},
        {"experiment": "Bounded baseline", "GPUs": 4, "requested": "2h", "reserved GPU-h": 8, "elapsed": "2h00m", "outcome": "Dataset construction deadline"},
        {"experiment": "Bounded T²MLR", "GPUs": 4, "requested": "2h", "reserved GPU-h": 8, "elapsed": "2h00m", "outcome": "Dataset construction deadline"},
    ]
    publication = {
        "paper": "arXiv:2607.15178",
        "backend": "OpenResearch Kubernetes",
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "maximum_concurrent_gpus": 16,
        "actual_wall_hours": 2.301211,
        "reserved_gpu_hours": 192,
        "paper_recurrence": "28→5 in a stated 32-layer model",
        "implemented_fallback": "24→5 in the pinned official 24-layer checkpoint",
    }
    return claim_rows, publication, run_rows


@app.cell
def _(mo):
    mo.md(r"""
    # T²MLR retrofit reproduction

    **Interactive publication companion for arXiv:2607.15178.** The requested
    quality and inference-overhead measurements were not reached. This is a
    blocked reproduction, not evidence that temporal middle-layer recurrence
    is ineffective.
    """)
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.stat(value="NO MEASUREMENT", label="Scientific result", caption="Stopped before model loading"),
            mo.stat(value="192 GPU-h", label="Reserved compute", caption="Hard aggregate cap"),
            mo.stat(value="2.301211 h", label="Actual wall time", caption="First launch to final terminal state"),
        ],
        widths="equal",
    )
    return


@app.cell
def _(claim_rows, mo):
    mo.vstack(
        [
            mo.md("## Claim-by-claim verdicts"),
            mo.ui.table(claim_rows, pagination=False, selection=None),
            mo.callout(
                mo.md(
                    "The required publication-schema verdict is **not-reproduced**. "
                    "Scientifically, both measurements are **not attempted** because "
                    "execution never reached a model forward pass."
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    evidence_view = mo.ui.radio(
        options=["Architecture", "Execution", "Compute ledger"],
        value="Architecture",
        label="Diagnostic view",
        inline=True,
    )
    evidence_view
    return (evidence_view,)


@app.cell
def _(evidence_view, mo, publication, run_rows):
    if evidence_view.value == "Architecture":
        diagnostic = mo.md(
            f"""
            ## Architecture substitution

            - **Paper:** {publication['paper_recurrence']}
            - **Pinned official checkpoint:** {publication['implemented_fallback']}

            The gated fusion and recurrent cache update were implemented, but a
            completed run would still be a partial architectural reproduction until
            the authors identify the 32-layer checkpoint used in the paper.
            """
        )
    elif evidence_view.value == "Execution":
        diagnostic = mo.md(
            """
            ## Where execution stopped

            `Kubernetes → NCCL/DDP → OpenMathReasoning filtering → ⛔`

            Both final logs emitted the paired run contract and a warning for a
            9,439-token candidate, then reached their two-hour Kubernetes deadlines.
            They emitted no `data_ready`, `model_ready`, `training_complete`,
            `eval_result`, or `inference_benchmark` event.
            """
        )
    else:
        diagnostic = mo.vstack(
            [
                mo.md("## Conservative reserved-compute ledger"),
                mo.ui.table(run_rows, pagination=False, selection=None),
            ]
        )
    diagnostic
    return


@app.cell
def _(mo, publication):
    mo.md(
        f"""
        ## Implemented mechanism

        ```python
        joined = torch.cat((current, recurrent), dim=-1)
        fused = current \\
            + tanh(gamma_current) * sigmoid(current_gate(joined)) * current \\
            + tanh(gamma_recurrent) * sigmoid(recurrent_gate(joined)) \\
              * recurrent_projection(recurrent)
        new_cache = rms_norm(h_end + previous_cache)
        ```

        The final paired protocol retained the full 1.7B checkpoint, used 512
        OpenMathReasoning examples, Jacobi depth 8/1, 256 GSM8K examples, 100
        MATH500 examples, and 128/512-token profiling lengths. The two branches
        differed only by the baseline/T²MLR variant flag.

        **Compute:** {publication['backend']}; {publication['gpu']}; maximum
        {publication['maximum_concurrent_gpus']} concurrent GPUs; actual wall time
        {publication['actual_wall_hours']} hours; {publication['reserved_gpu_hours']}
        total reserved GPU-hours.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    [Detailed report](https://github.com/rehaanahmad2013/t-2mlr-transformer-with-temporal-middle-layer-re/blob/main/reports/t2mlr-reproduction/report.md) · [Repository](https://github.com/rehaanahmad2013/t-2mlr-transformer-with-temporal-middle-layer-re)
    """)
    return


if __name__ == "__main__":
    app.run()
