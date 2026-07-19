#!/usr/bin/env python3
"""Paired, bounded reproduction of the T^2MLR retrofit and overhead claims.

All train/eval evidence is printed to stdout because local OpenResearch runs use
the run log as their evidence channel.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


CONFIG = json.loads(Path("experiment.json").read_text())
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
DEVICE = torch.device("cuda", LOCAL_RANK)
IS_MAIN = RANK == 0


def log(event: str, **values: Any) -> None:
    if IS_MAIN:
        print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed + RANK)
    torch.manual_seed(seed + RANK)
    torch.cuda.manual_seed_all(seed + RANK)


class TokenDataset(Dataset):
    def __init__(self, rows: list[dict[str, torch.Tensor]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


def right_shift(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)


def rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    work = x.float()
    work = work * torch.rsqrt(work.pow(2).mean(dim=-1, keepdim=True) + eps)
    return work.to(x.dtype)


class GatedFusion(nn.Module):
    """Equation 2.3 from arXiv:2607.15178, zero-gated at initialization."""

    def __init__(self, width: int):
        super().__init__()
        self.current_gate = nn.Linear(2 * width, width)
        self.recurrent_gate = nn.Linear(2 * width, width)
        self.recurrent_projection = nn.Linear(width, width, bias=False)
        self.gamma_current = nn.Parameter(torch.zeros(()))
        self.gamma_recurrent = nn.Parameter(torch.zeros(()))

    def forward(self, current: torch.Tensor, recurrent: torch.Tensor) -> torch.Tensor:
        joined = torch.cat((current, recurrent), dim=-1)
        current_term = (
            torch.tanh(self.gamma_current)
            * torch.sigmoid(self.current_gate(joined))
            * current
        )
        recurrent_term = (
            torch.tanh(self.gamma_recurrent)
            * torch.sigmoid(self.recurrent_gate(joined))
            * self.recurrent_projection(recurrent)
        )
        return current + current_term + recurrent_term


class TemporalMiddleLayerLM(nn.Module):
    """Retrofit wrapper using Jacobi cache refinement for teacher-forced SFT.

    The paper's named checkpoint is officially 24 layers, not the reported 32.
    Thus the configured 1-based endpoint is layer 24 (the last valid layer).
    To stay within the bounded reproduction budget, refinements are no-grad and
    the final fused pass carries gradients (backward depth one).
    """

    def __init__(self, base: nn.Module, start_layer: int, end_layer: int, depth: int):
        super().__init__()
        self.base = base
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.depth = depth
        layers = self.base.model.layers
        if not (1 <= start_layer <= end_layer <= len(layers)):
            raise ValueError(f"invalid 1-based recurrence range {start_layer}..{end_layer} for {len(layers)} layers")
        self.fusion = GatedFusion(base.config.hidden_size)
        self.eps = float(base.config.rms_norm_eps)

    @contextlib.contextmanager
    def _hooks(self, recurrent: torch.Tensor | None, capture: list[torch.Tensor] | None = None):
        handles = []
        if recurrent is not None:
            def inject(_module: nn.Module, args: tuple[Any, ...]):
                return (self.fusion(args[0], recurrent), *args[1:])
            handles.append(self.base.model.layers[self.start_layer - 1].register_forward_pre_hook(inject))
        if capture is not None:
            def save_end(_module: nn.Module, _args: tuple[Any, ...], output: Any):
                capture.append(output[0] if isinstance(output, tuple) else output)
            handles.append(self.base.model.layers[self.end_layer - 1].register_forward_hook(save_end))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _middle_state(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, recurrent: torch.Tensor | None) -> torch.Tensor:
        captured: list[torch.Tensor] = []
        with self._hooks(recurrent, captured):
            self.base.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        if len(captured) != 1:
            raise RuntimeError(f"expected one captured middle state, got {len(captured)}")
        return captured[0]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        with torch.no_grad():
            end_state = self._middle_state(input_ids, attention_mask, None)
            recurrent = right_shift(end_state)
            for _ in range(1, self.depth):
                end_state = self._middle_state(input_ids, attention_mask, recurrent)
                recurrent = rms_normalize(right_shift(end_state + recurrent), self.eps)
        with self._hooks(recurrent.detach()):
            return self.base(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False, return_dict=True)

    @torch.inference_mode()
    def exact_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
        recurrent: torch.Tensor | None,
    ) -> tuple[Any, torch.Tensor]:
        if recurrent is None:
            recurrent = torch.zeros(
                input_ids.shape[0], 1, self.base.config.hidden_size,
                device=input_ids.device, dtype=next(self.parameters()).dtype,
            )
        captured: list[torch.Tensor] = []
        with self._hooks(recurrent, captured):
            out = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        new_recurrent = rms_normalize(captured[0][:, -1:] + recurrent, self.eps)
        return out, new_recurrent


def build_train_rows(tokenizer: Any) -> tuple[list[dict[str, torch.Tensor]], str, int]:
    wanted = int(CONFIG["train_examples"])
    max_len = int(CONFIG["max_train_tokens"])
    stream = load_dataset(
        "nvidia/OpenMathReasoning",
        split="cot",
        streaming=True,
        revision=CONFIG["openmath_revision"],
    ).shuffle(seed=int(CONFIG["seed"]), buffer_size=1_024)
    rows: list[dict[str, torch.Tensor]] = []
    seen: set[str] = set()
    digest = hashlib.sha256()
    inspected = 0
    for item in stream:
        inspected += 1
        problem = str(item.get("problem", "")).strip()
        solution = str(item.get("generated_solution", "")).strip()
        source = str(item.get("problem_source", ""))
        if not problem or not solution or problem in seen or source == "MATH_training_set":
            continue
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        solution_ids = tokenizer(solution, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        ids = prompt_ids + solution_ids
        if len(ids) > max_len:
            continue
        labels = [-100] * len(prompt_ids) + solution_ids
        padding = max_len - len(ids)
        rows.append({
            "input_ids": torch.tensor(ids + [tokenizer.pad_token_id] * padding, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(ids) + [0] * padding, dtype=torch.long),
            "labels": torch.tensor(labels + [-100] * padding, dtype=torch.long),
        })
        seen.add(problem)
        digest.update(problem.encode("utf-8"))
        digest.update(b"\0")
        if len(rows) == wanted:
            break
    if len(rows) != wanted:
        raise RuntimeError(f"only found {len(rows)} eligible rows after inspecting {inspected}")
    return rows, digest.hexdigest(), inspected


def prepare_data(tokenizer: Any) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, str]], list[dict[str, str]]]:
    cache = Path("/tmp/t2mlr_reproduction_data.pt")
    if IS_MAIN:
        train_rows, fingerprint, inspected = build_train_rows(tokenizer)
        gsm = load_dataset("openai/gsm8k", "main", split="test", revision=CONFIG["gsm8k_revision"])
        math500 = load_dataset("HuggingFaceH4/MATH-500", split="test", revision=CONFIG["math500_revision"])
        gsm_limit = int(CONFIG["gsm8k_eval_examples"])
        math_limit = int(CONFIG["math500_eval_examples"])
        payload = {
            "train": train_rows,
            "gsm8k": [{"problem": x["question"], "answer": x["answer"].split("####")[-1].strip()} for x in gsm.select(range(gsm_limit))],
            "math500": [{"problem": x["problem"], "answer": x["answer"]} for x in math500.select(range(math_limit))],
            "fingerprint": fingerprint,
            "inspected": inspected,
        }
        torch.save(payload, cache)
        log("data_ready", train_examples=len(train_rows), gsm8k_examples=gsm_limit, math500_examples=math_limit,
            selected_problem_sha256=fingerprint, stream_rows_inspected=inspected)
    dist.barrier()
    payload = torch.load(cache, weights_only=False)
    return payload["train"], payload["gsm8k"], payload["math500"]


def unwrap_base(model: nn.Module) -> nn.Module:
    return model.base if isinstance(model, TemporalMiddleLayerLM) else model


def train(model: nn.Module, rows: list[dict[str, torch.Tensor]]) -> tuple[list[dict[str, float]], float]:
    sampler = DistributedSampler(rows, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, seed=int(CONFIG["seed"]))
    loader = DataLoader(TokenDataset(rows), batch_size=int(CONFIG["per_device_batch_size"]), sampler=sampler, num_workers=0)
    ddp = DDP(model, device_ids=[LOCAL_RANK], broadcast_buffers=False, gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(
        ddp.parameters(), lr=float(CONFIG["learning_rate"]), betas=(0.9, 0.999), weight_decay=float(CONFIG["weight_decay"])
    )
    accumulation = int(CONFIG["gradient_accumulation_steps"])
    total_steps = math.ceil(len(loader) / accumulation)
    warmup_steps = max(1, int(total_steps * float(CONFIG["warmup_ratio"])))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    curve: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    window_loss = torch.zeros((), device=DEVICE)
    optimizer_step = 0
    for batch_index, batch in enumerate(loader):
        batch = {key: value.to(DEVICE, non_blocking=True) for key, value in batch.items()}
        should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
        sync_context = contextlib.nullcontext() if should_step else ddp.no_sync()
        with sync_context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = ddp(**batch)
            loss = output.loss / accumulation
        loss.backward()
        window_loss += output.loss.detach()
        if should_step:
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            mean_loss = window_loss / accumulation
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= WORLD_SIZE
            point = {"step": float(optimizer_step), "loss": float(mean_loss.item()), "lr": float(scheduler.get_last_lr()[0])}
            curve.append(point)
            if IS_MAIN and (optimizer_step == 1 or optimizer_step % 10 == 0 or optimizer_step == total_steps):
                log("train_step", **point)
            window_loss.zero_()
    dist.barrier()
    elapsed = time.perf_counter() - started
    raw = ddp.module
    del ddp, optimizer, scheduler
    raw.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return curve, elapsed


def extract_boxed(text: str) -> str | None:
    starts = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    brace = text.find("{", start)
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1:index]
    return None


def normalize_answer(value: str) -> str:
    value = value.strip()
    boxed = extract_boxed(value)
    if boxed is not None:
        value = boxed
    value = re.sub(r"(?i).*?(?:final answer|answer is)\s*[:=]?", "", value).strip()
    value = value.replace("$", "").replace(",", "").replace("\\!", "")
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    value = value.replace(" ", "").rstrip(".。")
    if "=" in value and len(value.split("=")) == 2:
        value = value.split("=")[-1]
    return value.lower()


def candidate_answer(text: str) -> str:
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    matches = re.findall(r"(?i)(?:final answer|answer is)\s*[:=]?\s*([^\n]+)", text)
    if matches:
        return matches[-1]
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?", text)
    return numbers[-1] if numbers else text.splitlines()[-1] if text.splitlines() else ""


def answers_match(prediction: str, target: str) -> bool:
    p = normalize_answer(candidate_answer(prediction))
    t = normalize_answer(target)
    if p == t:
        return True
    try:
        return math.isclose(float(p), float(t), rel_tol=1e-6, abs_tol=1e-6)
    except ValueError:
        return False


@torch.inference_mode()
def generate_tokens(model: nn.Module, prompt_ids: list[int], max_new_tokens: int, stop_on_eos: bool) -> list[int]:
    base = unwrap_base(model)
    past = None
    recurrent = None
    logits = None
    total = 0
    for token in prompt_ids:
        total += 1
        ids = torch.tensor([[token]], device=DEVICE)
        mask = torch.ones((1, total), dtype=torch.long, device=DEVICE)
        if isinstance(model, TemporalMiddleLayerLM):
            out, recurrent = model.exact_step(ids, mask, past, recurrent)
        else:
            out = base(input_ids=ids, attention_mask=mask, past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1]
    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_token = int(torch.argmax(logits, dim=-1).item())
        generated.append(next_token)
        if stop_on_eos and next_token == base.config.eos_token_id:
            break
        total += 1
        ids = torch.tensor([[next_token]], device=DEVICE)
        mask = torch.ones((1, total), dtype=torch.long, device=DEVICE)
        if isinstance(model, TemporalMiddleLayerLM):
            out, recurrent = model.exact_step(ids, mask, past, recurrent)
        else:
            out = base(input_ids=ids, attention_mask=mask, past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1]
    return generated


def evaluate_dataset(model: nn.Module, tokenizer: Any, name: str, rows: list[dict[str, str]]) -> tuple[float, list[dict[str, Any]], float]:
    local: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index in range(RANK, len(rows), WORLD_SIZE):
        row = rows[index]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["problem"] + "\nSolve carefully and put the final answer in \\boxed{}."}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids[-384:]
        output_ids = generate_tokens(model, prompt_ids, int(CONFIG["max_new_tokens"]), stop_on_eos=True)
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        local.append({"index": index, "correct": answers_match(text, row["answer"]), "prediction": text, "target": row["answer"]})
        if len(local) % 25 == 0:
            print(json.dumps({"event": "eval_progress", "rank": RANK, "benchmark": name, "completed": len(local)}), flush=True)
    gathered: list[list[dict[str, Any]]] = [None for _ in range(WORLD_SIZE)]  # type: ignore[list-item]
    dist.all_gather_object(gathered, local)
    elapsed = time.perf_counter() - started
    merged = sorted((item for shard in gathered for item in shard), key=lambda x: x["index"])
    accuracy = sum(item["correct"] for item in merged) / len(merged)
    if IS_MAIN:
        for item in merged[:4]:
            log("eval_sample", benchmark=name, index=item["index"], correct=item["correct"], target=item["target"], prediction=item["prediction"][:1000])
        log("eval_result", benchmark=name, accuracy=accuracy, correct=sum(item["correct"] for item in merged), total=len(merged), elapsed_seconds=elapsed)
    return accuracy, merged, elapsed


def latency_and_memory(model: nn.Module) -> dict[str, Any]:
    base = unwrap_base(model)
    bos = int(base.config.bos_token_id)
    _ = generate_tokens(model, [bos], 32, stop_on_eos=False)
    torch.cuda.synchronize()
    static_bytes = torch.cuda.memory_allocated(DEVICE)
    measurements: dict[str, Any] = {}
    for length in CONFIG["latency_lengths"]:
        times: list[float] = []
        peaks: list[int] = []
        for _ in range(int(CONFIG["latency_repeats"])):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(DEVICE)
            started = time.perf_counter()
            _ = generate_tokens(model, [bos], int(length), stop_on_eos=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            peaks.append(torch.cuda.max_memory_allocated(DEVICE))
        measurements[str(length)] = {"seconds": statistics.median(times), "peak_bytes": int(statistics.median(peaks))}
    measurements["static_bytes"] = static_bytes
    all_measurements: list[dict[str, Any]] = [None for _ in range(WORLD_SIZE)]  # type: ignore[list-item]
    dist.all_gather_object(all_measurements, measurements)
    if IS_MAIN:
        aggregate: dict[str, Any] = {"static_bytes": int(statistics.median(x["static_bytes"] for x in all_measurements))}
        for length in CONFIG["latency_lengths"]:
            key = str(length)
            aggregate[key] = {
                "seconds": statistics.median(x[key]["seconds"] for x in all_measurements),
                "peak_bytes": int(statistics.median(x[key]["peak_bytes"] for x in all_measurements)),
            }
        log("inference_benchmark", measurements=aggregate)
        return aggregate
    return {}


def main() -> None:
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group("nccl", device_id=DEVICE)
    seed_everything(int(CONFIG["seed"]))
    job_started = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(DEVICE)
    log(
        "run_contract",
        config=CONFIG,
        backend="OpenResearch Kubernetes",
        gpu_model=gpu_name,
        gpu_count=WORLD_SIZE,
        reserved_gpu_hours=float(CONFIG["requested_gpus"]) * float(CONFIG["requested_wall_hours"]),
        planned_aggregate_reserved_gpu_hours=CONFIG["planned_aggregate_reserved_gpu_hours"],
        torch_version=torch.__version__,
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_id"], revision=CONFIG["model_revision"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    train_rows, gsm8k_rows, math500_rows = prepare_data(tokenizer)
    base = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_id"], revision=CONFIG["model_revision"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(DEVICE)
    if base.config.num_hidden_layers != 24:
        raise RuntimeError(f"pinned checkpoint layer count changed: {base.config.num_hidden_layers}")
    base.config.use_cache = False
    if CONFIG["variant"] == "t2mlr":
        model: nn.Module = TemporalMiddleLayerLM(
            base,
            int(CONFIG["recurrence_start_layer"]),
            int(CONFIG["recurrence_end_layer"]),
            int(CONFIG["jacobi_forward_depth"]),
        ).to(DEVICE)
    elif CONFIG["variant"] == "baseline":
        model = base
    else:
        raise ValueError(f"unknown variant {CONFIG['variant']}")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    log(
        "model_ready",
        official_layers=base.config.num_hidden_layers,
        paper_reported_layers=32,
        paper_recurrence="5..28",
        implemented_recurrence=(f"{CONFIG['recurrence_start_layer']}..{CONFIG['recurrence_end_layer']}" if CONFIG["variant"] == "t2mlr" else None),
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
    )
    curve, train_seconds = train(model, train_rows)
    log(
        "training_complete",
        elapsed_seconds=train_seconds,
        optimizer_steps=len(curve),
        final_loss=curve[-1]["loss"],
        gamma_current=float(model.fusion.gamma_current.detach().float().item()) if isinstance(model, TemporalMiddleLayerLM) else None,
        gamma_recurrent=float(model.fusion.gamma_recurrent.detach().float().item()) if isinstance(model, TemporalMiddleLayerLM) else None,
    )
    base = unwrap_base(model)
    base.config.use_cache = True
    model.eval()
    gsm8k_accuracy, _, gsm_seconds = evaluate_dataset(model, tokenizer, "GSM8K", gsm8k_rows)
    math500_accuracy, _, math_seconds = evaluate_dataset(model, tokenizer, "MATH-500", math500_rows)
    overhead = latency_and_memory(model)
    total_elapsed = time.perf_counter() - job_started
    if IS_MAIN:
        result = {
            "variant": CONFIG["variant"],
            "gsm8k_accuracy": gsm8k_accuracy,
            "math500_accuracy": math500_accuracy,
            "training_seconds": train_seconds,
            "gsm8k_eval_seconds": gsm_seconds,
            "math500_eval_seconds": math_seconds,
            "total_elapsed_seconds": total_elapsed,
            "inference": overhead,
            "backend": "OpenResearch Kubernetes",
            "gpu_model": gpu_name,
            "gpu_count": WORLD_SIZE,
            "reserved_gpu_hours": float(CONFIG["requested_gpus"]) * float(CONFIG["requested_wall_hours"]),
            "planned_aggregate_reserved_gpu_hours": CONFIG["planned_aggregate_reserved_gpu_hours"],
        }
        print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
