# Copyright 2026 The LatentForge Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""What the phase trainers guarantee about a model handed to them already assembled."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from conftest import BASE_VOCAB_SIZE
from polyplex_research.adapters.phase_1_efficient_adapter import (
    EfficientAdapterP1Config,
    EfficientAdapterP1Model,
)
from polyplex_research.utils.phase import Phase1Trainer, load_english_states


def test_prepare_freezes_a_hand_built_adapter(base_model):
    r"""
    A model built outside `assemble` has never been frozen, and the trainer is the only thing standing
    between that and a run that quietly trains all 7B.
    """
    config = EfficientAdapterP1Config.from_base_model(base_model, base_vocab_size=BASE_VOCAB_SIZE)
    adapter = EfficientAdapterP1Model(base_model, config)
    assert any(p.requires_grad for p in adapter.model.parameters())

    Phase1Trainer._prepare(Phase1Trainer, adapter, None)

    trainable = {id(p) for p in adapter.adapter_parameters()}
    assert all(not p.requires_grad or id(p) in trainable for p in adapter.parameters())
    assert trainable and all(p.requires_grad for p in adapter.adapter_parameters())


# ---------------------------------------------------------------------------
# Keeping English in the objective
# ---------------------------------------------------------------------------


def build_adapter(model, **kwargs):
    config = EfficientAdapterP1Config.from_base_model(
        model, base_vocab_size=BASE_VOCAB_SIZE, **kwargs
    )
    adapter = EfficientAdapterP1Model(model, config)
    adapter.freeze_base()
    return adapter


def english_cache(model, frames: int = 16):
    r"""
    Args:
        model (`PreTrainedModel`):
            The model whose hidden size the states are shaped to.
        frames (`int`, *optional*, defaults to 16):
            How many frames the cache holds.

    Returns:
        `tuple`: States and labels of a cache the preservation term can be scored on.
    """
    torch.manual_seed(3)
    return (
        torch.randn(frames, model.config.hidden_size, dtype=torch.float16),
        torch.randint(0, BASE_VOCAB_SIZE, (frames,)),
    )


def stub_trainer(adapter, english_lambda, states=None, labels=None, english_frames=8):
    r"""
    Returns:
        `SimpleNamespace`: The attributes [`Phase1Trainer.compute_loss`] reads, so the objective can be
        exercised without standing up a training loop around it. `sync_gradients` is off because no step
        boundary is being crossed, which is the only thing the logging branch reads.
    """
    stub = SimpleNamespace(
        model=adapter,
        english_lambda=english_lambda,
        english_frames=english_frames,
        english_states=states,
        english_labels=labels,
        timing_lambda=0.0,
        leak_lambda=0.0,
        timing_kl_lambda=0.0,
        english_guard_path=None,
        english_guard_lambda=0.0,
        text_weight=0.0,
        _text_loader=None,
        accelerator=SimpleNamespace(sync_gradients=False, unwrap_model=lambda model, **kwargs: model),
    )
    # The trainer's own term, reading nothing but the attributes above.
    stub.english_penalty = lambda: Phase1Trainer.english_penalty(stub)
    return stub


@pytest.mark.parametrize("mixture", [0, 4], ids=["free_rows", "row_mixture"])
def test_zero_weight_leaves_the_loss_bit_identical(base_model, text_batch, mixture):
    r"""
    The option has to be free to exist: a run that does not ask for the term must compute what it
    computed before there was one, down to the bit.
    """
    adapter = build_adapter(base_model, base_row_mixture=mixture)
    expected = adapter(**text_batch).loss

    states, labels = english_cache(base_model)
    loss = Phase1Trainer.compute_loss(
        stub_trainer(adapter, 0.0, states, labels), adapter, dict(text_batch)
    )
    assert torch.equal(loss, expected)


@pytest.mark.parametrize("mixture", [0, 4], ids=["free_rows", "row_mixture"])
def test_a_nonzero_weight_changes_the_loss(base_model, text_batch, mixture):
    adapter = build_adapter(base_model, base_row_mixture=mixture)
    states, labels = english_cache(base_model)

    plain = adapter(**text_batch).loss
    penalised = Phase1Trainer.compute_loss(
        stub_trainer(adapter, 3.0, states, labels), adapter, dict(text_batch)
    )
    assert not torch.equal(penalised, plain)
    assert penalised.item() > plain.item()


@pytest.mark.parametrize("mixture", [0, 4], ids=["free_rows", "row_mixture"])
def test_the_term_reaches_the_trained_parameters_and_nothing_else(base_model, text_batch, mixture):
    r"""
    The term is only sound if it is a function of what the phase trains. The pretrained rows are what it
    holds English fixed against, so a step driven by the term alone has to leave them bit identical
    while still moving the appended ones.
    """
    adapter = build_adapter(base_model, base_row_mixture=mixture)
    frozen = base_model.lm_head.weight.data[:BASE_VOCAB_SIZE].clone()
    appended = base_model.lm_head.weight.data[BASE_VOCAB_SIZE:].clone()

    # Nothing Korean is scored, so the whole gradient is the one the English term supplies.
    batch = dict(text_batch)
    batch["text_labels"] = torch.full_like(batch["text_labels"], -100)

    states, labels = english_cache(base_model)
    # Decay reads no gradient and would shrink the pretrained rows of a table whatever the objective
    # says, which is why `Phase1Trainer.create_optimizer` exempts the tables from it.
    optimizer = torch.optim.AdamW(adapter.adapter_parameters(), lr=0.1, weight_decay=0.0)
    loss = Phase1Trainer.compute_loss(stub_trainer(adapter, 3.0, states, labels), adapter, batch)
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in adapter.adapter_parameters())
    optimizer.step()
    adapter.materialize_rows()

    assert torch.equal(base_model.lm_head.weight.data[:BASE_VOCAB_SIZE], frozen)
    assert not torch.equal(base_model.lm_head.weight.data[BASE_VOCAB_SIZE:], appended)


def test_the_cache_loader_rejects_states_and_labels_of_different_length(tmp_path):
    r"""
    A state and its label are one frame read twice. Nothing downstream notices when they are paired
    wrongly, the term simply scores a distribution nobody measured, so the length is checked on load.
    """
    path = tmp_path / "english_states.safetensors"
    save_file(
        {"states": torch.randn(8, 32, dtype=torch.float16), "labels": torch.zeros(7, dtype=torch.long)},
        path,
    )
    with pytest.raises(ValueError, match="8 states and 7 labels"):
        load_english_states(str(path))


def test_the_cache_loader_returns_what_was_written(tmp_path):
    path = tmp_path / "english_states.safetensors"
    states = torch.randn(8, 32, dtype=torch.float16)
    labels = torch.randint(0, BASE_VOCAB_SIZE, (8,))
    save_file({"states": states, "labels": labels}, path)

    loaded_states, loaded_labels = load_english_states(str(path))
    assert torch.equal(loaded_states, states)
    assert torch.equal(loaded_labels, labels)
    assert loaded_labels.dtype == torch.long


def test_the_device_spec_names_a_provider_and_an_accelerator(tmp_path):
    import letify

    from polyplex_research.utils.phase import resolve_device

    # An empty project directory, so neither the developer's home nor the repository's .letify is read.
    launcher = letify.Launcher(config=tmp_path, home=False)
    assert resolve_device(launcher, "local.CPU") == launcher.providers.local.CPU


def test_a_bare_provider_takes_its_first_accelerator(tmp_path):
    import letify

    from polyplex_research.utils.phase import resolve_device

    launcher = letify.Launcher(config=tmp_path, home=False)
    devices = [name for name in launcher.providers.local.discover() if name != "CPU"] or ["CPU"]
    assert resolve_device(launcher, "local") == getattr(launcher.providers.local, devices[0])


def test_an_unknown_accelerator_is_refused(tmp_path):
    import letify

    from polyplex_research.utils.phase import resolve_device

    with pytest.raises(Exception):
        resolve_device(letify.Launcher(config=tmp_path, home=False), "local.NO_SUCH_CARD")


def test_dense_input_files_go_through_the_drive_store_and_come_back(tmp_path, monkeypatch):
    import shutil

    import letify

    from polyplex_research.utils.phase import fetch_input_file, stash_input_files

    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(letify, "stash", lambda name, path: shutil.copy(path, store / name), raising=False)
    monkeypatch.setattr(
        letify, "unstash", lambda name, path: shutil.copy(store / name, path) if (store / name).exists() else None,
        raising=False,
    )
    alignment = tmp_path / "alignment.safetensors"
    alignment.write_bytes(b"rotation")
    args = ["--steps", "5", "--alignment", str(alignment), "--run-name", "r"]
    sent = stash_input_files(args, "r")
    assert sent == ["--steps", "5", "--alignment", "store:r-alignment", "--run-name", "r"]
    assert stash_input_files(sent, "r") == sent
    fetched = fetch_input_file("store:r-alignment", str(tmp_path / "inputs"))
    assert open(fetched, "rb").read() == b"rotation"
    assert fetch_input_file(str(alignment), str(tmp_path)) == str(alignment)
    assert fetch_input_file(None, str(tmp_path)) is None
    with pytest.raises(FileNotFoundError):
        fetch_input_file("store:missing", str(tmp_path / "inputs"))


def test_a_checkpoint_file_goes_to_hugging_face_and_drive_and_comes_back_from_either(tmp_path, monkeypatch):
    import shutil

    import letify

    import polyplex_research.utils.phase as phase

    hub, drive = tmp_path / "hub", tmp_path / "drive"
    hub.mkdir()
    drive.mkdir()
    monkeypatch.setattr(phase, "upload_checkpoint_file", lambda run, path: shutil.copy(path, hub / path.name))
    monkeypatch.setattr(phase, "download_checkpoint_file", lambda run, name, directory: shutil.copy(hub / name, directory / name))
    monkeypatch.setattr(letify, "stash", lambda name, path: shutil.copy(path, drive / name), raising=False)
    monkeypatch.setattr(
        letify, "unstash", lambda name, path: shutil.copy(drive / name, path) if (drive / name).exists() else None,
        raising=False,
    )
    trained, masters, optimizer = _trained_state()
    phase.write_checkpoint(tmp_path / "ckpt", 7, trained, masters, optimizer, {},
                           lambda path: phase.stash_checkpoint_file("r", path))
    assert not list((tmp_path / "ckpt").iterdir())
    assert sorted(item.name for item in hub.iterdir())[-1] == "manifest.json"
    assert "r-checkpoint" in {item.name for item in drive.iterdir()}
    assert phase.read_checkpoint(phase.unstash_checkpoint("r", tmp_path / "a"), tensors=False)["step"] == 7
    shutil.rmtree(drive)
    drive.mkdir()
    assert phase.read_checkpoint(phase.unstash_checkpoint("r", tmp_path / "b"), tensors=False)["step"] == 7


def test_a_failed_drive_handoff_sends_the_rest_of_the_checkpoint_to_hugging_face_alone(tmp_path, monkeypatch):
    import letify

    import polyplex_research.utils.phase as phase

    uploaded, asked = [], []
    monkeypatch.setattr(phase, "upload_checkpoint_file", lambda run, path: uploaded.append(path.name))

    def stash(name, path):
        asked.append(name)
        raise TimeoutError("no answer within 600 s")

    monkeypatch.setattr(letify, "stash", stash, raising=False)
    for name in ("checkpoint-00001.safetensors", "checkpoint-00002.safetensors", "manifest.json"):
        (tmp_path / name).write_text("x")
        phase.stash_checkpoint_file("r", tmp_path / name)
    assert uploaded == ["checkpoint-00001.safetensors", "checkpoint-00002.safetensors", "manifest.json"]
    assert asked == ["r-checkpoint-checkpoint-00001"]
    assert not list(tmp_path.iterdir())
    (tmp_path / "checkpoint-00001.safetensors").write_text("x")
    phase.stash_checkpoint_file("r", tmp_path / "checkpoint-00001.safetensors")
    assert len(asked) == 2


def test_the_reference_key_follows_the_model_and_the_evaluation_texts():
    from polyplex_research.utils.phase import reference_key

    held_out = {"en": ["a text"], "ko": ["한국어"]}
    assert reference_key("Qwen/Qwen3-8B", held_out) == reference_key("Qwen/Qwen3-8B", dict(held_out))
    assert reference_key("Qwen/Qwen3-8B", held_out) != reference_key("Qwen/Qwen3-4B", held_out)
    assert reference_key("Qwen/Qwen3-8B", held_out) != reference_key("Qwen/Qwen3-8B", {"en": ["other"], "ko": ["한국어"]})


def test_a_stored_reference_reads_back(monkeypatch, tmp_path):
    import huggingface_hub

    import polyplex_research.utils.phase as phase

    files = {}

    class Api:
        def create_repo(self, *args, **kwargs):
            pass

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, commit_message):
            files[path_in_repo] = path_or_fileobj

    def download(repo_id, filename):
        if filename not in files:
            raise FileNotFoundError(filename)
        path = tmp_path / "ref.json"
        path.write_bytes(files[filename])
        return str(path)

    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert phase.fetch_reference("abc") is None
    phase.store_reference("abc", {"en": 0.744})
    assert phase.fetch_reference("abc") == {"en": 0.744}


def test_a_reference_is_read_from_the_local_disk_before_the_hub(monkeypatch, tmp_path):
    import json

    import huggingface_hub

    import polyplex_research.utils.phase as phase

    downloads = []

    def download(repo_id, filename):
        downloads.append(filename)
        path = tmp_path / "hub.json"
        path.write_text(json.dumps({"en": 0.5}))
        return str(path)

    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert phase.fetch_reference("abc") == {"en": 0.5}
    assert json.loads((tmp_path / "home" / "reference" / "abc.json").read_text()) == {"en": 0.5}
    assert phase.fetch_reference("abc") == {"en": 0.5}
    assert downloads == ["reference/abc.json"]


def test_a_zero_appended_output_projection_gets_the_text_llm_one_back_and_a_fresh_optimizer_state(monkeypatch):
    import polyplex_research.utils.phase as phase

    down = "model.layers.2.mlp.language_mlp.down_proj.weight"
    weights = {
        "model.layers.0.mlp.language_norm.weight": torch.ones(4),
        "model.layers.2.mlp.language_mlp.up_proj.weight": torch.randn(6, 4),
        down: torch.zeros(4, 6, dtype=torch.bfloat16),
        "model.layers.2.mlp.language_scale": torch.zeros(4),
    }
    state = {
        "weights": weights,
        "optimizer": {"muon": {"state": {0: {"m": torch.ones(1)}, 1: {"m": torch.ones(1)}}}, "adamw": {"state": {0: {}, 1: {}}}},
    }
    source = torch.randn(4, 6)
    asked = []
    monkeypatch.setattr(phase, "language_model_tensor", lambda model, name: asked.append(name) or source)
    restored = phase.restore_appended_language_output(state, "Qwen/Qwen3-8B", [2], [0, 1, 35])
    assert restored == [down] and asked == ["model.layers.35.mlp.down_proj.weight"]
    assert state["weights"][down].dtype == torch.bfloat16
    torch.testing.assert_close(state["weights"][down], source.to(torch.bfloat16), rtol=0, atol=0)
    # The projection is the second two dimensional weight, so its Muon state goes; the rest stay.
    assert list(state["optimizer"]["muon"]["state"]) == [0] and list(state["optimizer"]["adamw"]["state"]) == [0, 1]
    assert phase.restore_appended_language_output(state, "Qwen/Qwen3-8B", [2], [0, 1, 35]) == []


def test_a_resume_takes_the_store_with_the_later_step(tmp_path, monkeypatch):
    import shutil

    import letify

    import polyplex_research.utils.phase as phase

    trained, masters, optimizer = _trained_state()
    hub, drive = tmp_path / "hub", tmp_path / "drive"
    phase.write_checkpoint(hub, 1000, trained, masters, optimizer, {})
    phase.write_checkpoint(tmp_path / "old", 500, trained, masters, optimizer, {})
    drive.mkdir()
    shutil.copy(tmp_path / "old" / "manifest.json", drive / "r-checkpoint")
    shutil.copy(tmp_path / "old" / "checkpoint.safetensors", drive / "r-checkpoint-checkpoint")
    monkeypatch.setattr(phase, "download_checkpoint_file", lambda run, name, directory: shutil.copy(hub / name, directory / name))
    monkeypatch.setattr(
        letify, "unstash", lambda name, path: shutil.copy(drive / name, path) if (drive / name).exists() else None,
        raising=False,
    )
    fetched = phase.unstash_checkpoint("r", tmp_path / "fetch")
    assert phase.read_checkpoint(fetched, tensors=False)["step"] == 1000
    assert sorted(item.name for item in fetched.iterdir()) == ["checkpoint.safetensors", "manifest.json"]
    shutil.rmtree(hub)
    hub.mkdir()
    assert phase.read_checkpoint(phase.unstash_checkpoint("r", tmp_path / "fetch2"), tensors=False)["step"] == 500


def test_a_checkpoint_file_from_the_hub_lands_once_in_the_directory(tmp_path, monkeypatch):
    import os

    import huggingface_hub

    import polyplex_research.utils.phase as phase

    def download(repo_id, filename, local_dir):
        path = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write("tensors")
        return path

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    target = tmp_path / "resume"
    target.mkdir()
    phase.download_checkpoint_file("r", "checkpoint.safetensors", target)
    assert sorted(item.name for item in target.iterdir()) == ["checkpoint.safetensors"]
    assert (target / "checkpoint.safetensors").read_text() == "tensors"


def test_english_states_keep_the_scored_frames_and_read_back(tmp_path):
    from conftest import AUDIO_VOCAB_SIZE, NUM_CODEBOOKS, build_model
    from safetensors.torch import save_file

    from polyplex_research.utils.phase import collect_english_states

    model = build_model(seed=0).eval()
    frames = 6
    labels = torch.full((2, frames), -100)
    labels[0, 2] = 5
    labels[1, 3:5] = 7
    batch = {
        "input_ids": torch.randint(0, BASE_VOCAB_SIZE, (2, frames)),
        "attention_mask": torch.ones(2, frames, dtype=torch.bool),
        "assistant_audio_codes": torch.randint(0, AUDIO_VOCAB_SIZE, (2, NUM_CODEBOOKS, frames)),
        "user_audio_codes": torch.randint(0, AUDIO_VOCAB_SIZE, (2, NUM_CODEBOOKS, frames)),
        "text_labels": labels,
    }
    states, kept = collect_english_states(model, lambda items: batch, [None], batch_size=1, device="cpu")

    # Position t carries the label of position t + 1, so the three labelled frames become three states.
    assert kept.tolist() == [5, 7, 7]
    assert states.shape == (3, model.config.hidden_size)
    save_file({"states": states.contiguous(), "labels": kept.contiguous()}, tmp_path / "cache.safetensors")
    loaded_states, loaded_labels = load_english_states(str(tmp_path / "cache.safetensors"))
    assert torch.equal(loaded_labels, kept) and torch.equal(loaded_states, states)


class SilentProcessor:
    r"""Stands in for the processor where only the silence is read."""

    def get_silence_audio_codes(self, frames, batch_size=1):
        return torch.ones(batch_size, 4, frames, dtype=torch.long)


def _dense_trainer(base_model, tmp_path, **kwargs):
    from transformers import TrainingArguments

    config = EfficientAdapterP1Config.from_base_model(base_model, base_vocab_size=BASE_VOCAB_SIZE)
    adapter = EfficientAdapterP1Model(base_model, config)
    rows = [{"segments": [torch.tensor([5, 6, 7]), torch.tensor([8, 9])]}, {"segments": [torch.tensor([10, 11])]}]
    return Phase1Trainer(
        model=adapter,
        args=TrainingArguments(output_dir=str(tmp_path), per_device_train_batch_size=2, report_to=[], max_steps=1),
        train_dataset=rows,
        dense=True,
        processor=SilentProcessor(),
        **kwargs,
    )


def test_a_dense_trainer_scores_packed_rows_and_reaches_the_adapter(base_model, tmp_path):
    trainer = _dense_trainer(base_model, tmp_path)
    batch = trainer.data_collator(trainer.train_dataset)
    assert "position_ids" in batch and "attention_mask" not in batch
    batch = trainer._prepare_inputs(batch)
    trainer.model.train()
    loss = trainer.compute_loss(trainer.model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.model.adapter_parameters())


def test_a_logged_training_step_reports_the_korean_loss_by_vocabulary_half(base_model, tmp_path):
    r"""
    The training loss of a run with English preservation is the Korean loss plus the English term, so
    the Korean loss is reported on its own, over the whole text and over the appended and the pretrained
    halves of the vocabulary.
    """
    trainer = _dense_trainer(base_model, tmp_path)
    trainer.train_dataset = [{"segments": [torch.tensor([5, 50, 7, 55])]}, {"segments": [torch.tensor([60, 9, 52])]}]
    batch = trainer._prepare_inputs(trainer.data_collator(trainer.train_dataset))
    trainer.model.train()
    trainer.args.logging_steps = 1
    trainer.accelerator.gradient_state._set_sync_gradients(True)
    logged = []
    trainer.log = logged.append

    loss = trainer.compute_loss(trainer.model, batch)

    with torch.no_grad():
        logits = trainer.model(**batch).logits[:, :-1].float()
    labels = batch["text_labels"][:, 1:]
    scored = labels != -100
    losses = torch.nn.functional.cross_entropy(logits[scored], labels[scored], reduction="none")
    appended = labels[scored] >= BASE_VOCAB_SIZE
    assert len(logged) == 1
    metrics = logged[0]
    assert metrics["korean_loss"] == pytest.approx(loss.item(), rel=1e-5)
    assert metrics["korean_new_loss"] == pytest.approx(losses[appended].mean().item(), rel=1e-5)
    assert metrics["korean_base_loss"] == pytest.approx(losses[~appended].mean().item(), rel=1e-5)


def test_an_unlogged_training_step_computes_no_korean_metric(base_model, tmp_path):
    trainer = _dense_trainer(base_model, tmp_path)
    batch = trainer._prepare_inputs(trainer.data_collator(trainer.train_dataset))
    trainer.model.train()
    trainer.args.logging_steps = 2
    trainer.state.global_step = 1
    trainer.accelerator.gradient_state._set_sync_gradients(True)
    logged = []
    trainer.log = logged.append

    trainer.compute_loss(trainer.model, batch)
    assert logged == []


def test_a_dense_trainer_refuses_terms_that_read_duplex_frames(base_model, tmp_path):
    with pytest.raises(ValueError, match="dense"):
        _dense_trainer(base_model, tmp_path, timing_kl_lambda=1.0)


def test_a_torchrun_worker_runs_main_itself_rather_than_through_letify(monkeypatch):
    import polyplex_research.utils.phase as phase

    called = []
    monkeypatch.setattr(phase, "main", lambda args=None: called.append(args))
    monkeypatch.setenv("WORLD_SIZE", "2")
    phase.launch(["--phase", "1"])
    assert called == [["--phase", "1"]]


def test_loss_terms_reach_the_adapter_through_a_data_parallel_wrapper(base_model, tmp_path):
    trainer = _dense_trainer(base_model, tmp_path)

    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, **kwargs):
            return self.module(**kwargs)

    trainer.accelerator.unwrap_model = lambda model, **kwargs: model.module if isinstance(model, Wrapper) else model
    trainer.leak_lambda = 1.0
    trainer.model.confined_leak = lambda: torch.zeros(())
    batch = {k: v.to(trainer.args.device) for k, v in trainer.data_collator(trainer.train_dataset).items()}
    wrapped = Wrapper(trainer.model).train()
    loss = trainer.compute_loss(wrapped, batch)
    assert torch.isfinite(loss)


def test_fixed_rows_hand_the_optimizer_only_the_session_expert(base_model, tmp_path):
    from transformers import TrainingArguments

    config = EfficientAdapterP1Config.from_base_model(
        base_model,
        base_vocab_size=BASE_VOCAB_SIZE,
        base_row_mixture=4,
        session_expert_layers="0-0",
        session_expert_rank=2,
        train_rows=False,
    )
    adapter = EfficientAdapterP1Model(base_model, config)
    trainer = Phase1Trainer(
        model=adapter,
        args=TrainingArguments(output_dir=str(tmp_path), per_device_train_batch_size=2, report_to=[], max_steps=1),
        train_dataset=[{"segments": [torch.tensor([5, 6, 7])]}],
        dense=True,
        processor=SilentProcessor(),
    )
    stepped = {id(p) for group in trainer.create_optimizer().param_groups for p in group["params"]}
    expert = adapter.confined[0]
    assert stepped == {id(expert.up), id(expert.basis)}


def test_the_run_record_carries_every_phase_option_the_data_sizes_and_the_commit():
    from polyplex_research.utils.phase import PhaseArguments, run_record

    phase_args = PhaseArguments(phase=1, model_name_or_path="base", dataset_path="data", train_rows=False)
    record = run_record(phase_args, train_size=10, eval_size=3, text_size=None)
    assert record["phase/train_rows"] is False
    assert record["phase/dataset_path"] == "data"
    assert record["record/train_data_size"] == 10 and record["record/eval_data_size"] == 3
    assert "record/text_data_size" not in record
    assert len(record["record/git_commit"]) == 40


def _dense_rows():
    return [
        {"segments": [torch.tensor([5, 6, 7, 8]), torch.tensor([9, 10, 11])]},
        {"segments": [torch.tensor([12, 13, 14])]},
    ]


def test_the_dense_score_leaves_out_the_first_token_of_every_document():
    from polyplex_research.utils.phase import segment_cross_entropy

    torch.manual_seed(0)
    logits = torch.randn(1, 6, 20)
    targets = torch.tensor([[-100, 3, 4, 5, -100, 7]])
    positions = torch.tensor([[0, 1, 2, 3, 0, 1]])
    total, count = segment_cross_entropy(logits, targets, positions, first_position=1)
    # Only positions 2 and 3 are scored: 1 and 5 open a document, 0 and 4 are opening frames.
    expected = torch.nn.functional.cross_entropy(logits[0, 1:3], targets[0, 2:4], reduction="sum")
    assert count == 2
    assert total == pytest.approx(expected.item(), rel=1e-6)


def test_the_text_llm_reads_the_same_documents_without_opening_frames_or_filler():
    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.phase import language_model_view

    from conftest import build_polyplex

    model = build_polyplex()
    rows = _dense_rows()
    batch = dense_collate_fn(SilentProcessor(), model)(rows)
    ids, positions = language_model_view(batch, 1)
    assert ids.tolist() == [[12, 13, 14]]
    assert positions.tolist() == [[0, 1, 2]]
    ids, positions = language_model_view(batch, 0)
    assert ids.tolist() == [[5, 6, 7, 8, 9, 10, 11]]
    assert positions.tolist() == [[0, 1, 2, 3, 0, 1, 2]]


def test_the_dense_check_trains_only_the_language_side_normalizations():
    from polyplex_research.utils.phase import dense_trainable_parameters

    from conftest import build_polyplex

    model = build_polyplex()
    names = dense_trainable_parameters(model)
    assert sorted(names) == ["model.layers.0.mlp.language_norm.weight", "model.layers.1.mlp.language_norm.weight", "model.text_norm.weight"]
    assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == set(names)


def test_a_dense_run_moves_the_normalizations_and_reports_the_gap_to_the_text_llm():
    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.phase import dense_trainable_parameters, train_dense

    from conftest import build_polyplex

    model = build_polyplex().to(torch.bfloat16)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters()}
    trained = dense_trainable_parameters(model)
    logged = []
    result = train_dense(
        model,
        dense_collate_fn(SilentProcessor(), model),
        train_dataset=_dense_rows() * 6,
        eval_sets={"ko": _dense_rows()},
        reference={"ko": 2.0},
        steps=3,
        batch_size=4,
        micro_batch_size=2,
        learning_rate=1e-2,
        warmup_steps=1,
        eval_every=2,
        log=logged.append,
    )
    for name, parameter in model.named_parameters():
        moved = not torch.equal(parameter, frozen[name])
        assert moved == (name in trained), name
    assert [entry["step"] for entry in result["evaluations"]] == [0, 2, 3]
    final = result["evaluations"][-1]
    assert final["eval/ko_gap"] == pytest.approx(final["eval/ko_loss"] - 2.0)
    assert sum("train/loss" in entry for entry in logged) == 3


def test_a_dense_run_stops_when_its_rows_run_out():
    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.phase import train_dense

    from conftest import build_polyplex

    model = build_polyplex()
    result = train_dense(
        model,
        dense_collate_fn(SilentProcessor(), model),
        train_dataset=_dense_rows() * 2,
        eval_sets={},
        reference={},
        steps=10,
        batch_size=2,
        micro_batch_size=2,
        learning_rate=1e-3,
        warmup_steps=0,
        eval_every=100,
    )
    assert result["steps"] == 2


def test_the_dense_command_line_reaches_the_dense_check(monkeypatch):
    import polyplex_research.utils.phase as phase

    called = {}
    monkeypatch.setattr(phase, "run_dense_adaptation", lambda **kwargs: called.update(kwargs) or {"steps": 1})
    result = phase.dense_main(["--steps", "5", "--eval-rows", "2", "--language-ffn-layers", "0,2", "--run-name", "x"])
    assert result == {"steps": 1}
    assert called["steps"] == 5 and called["eval_rows"] == 2 and called["run_name"] == "x"
    assert called["language_ffn_layers"] == [0, 2]
    assert called["batch_size"] == 24 and called["learning_rate"] == 1e-3
    assert called["quantization"] == "nvfp4"
    assert called["backward_dtype"] == "bf16"
    phase.dense_main(["--backward-dtype", "mxfp8"])
    assert called["backward_dtype"] == "mxfp8"
    phase.dense_main(["--alignment", "a.safetensors", "--evaluate-at-start"])
    assert called["alignment"] == "a.safetensors" and called["evaluate_at_start"] is True


def test_a_dense_run_reads_text_alone_in_full_nvfp4_unless_told_otherwise(monkeypatch):
    import polyplex_research.utils.phase as phase

    called = {}
    monkeypatch.setattr(phase, "run_dense_adaptation", lambda **kwargs: called.update(kwargs) or {"steps": 1})
    phase.dense_main([])
    assert called["full_precision_layers"] == () and called["silent_audio"] is False
    assert "speech_items" not in called
    phase.dense_main(["--silent-audio", "--full-precision-layers", "0,1"])
    assert called["full_precision_layers"] == (0, 1) and called["silent_audio"] is True


def test_a_dense_result_is_saved_as_its_metrics(tmp_path):
    import json

    from polyplex_research.utils.phase import save_dense_result

    result = {"steps": 2, "evaluations": [{"step": 0, "eval/ko_loss": 3.0}], "wandb": None, "checkpoint": "run-checkpoint"}
    save_dense_result(result, str(tmp_path))
    assert json.loads((tmp_path / "result.json").read_text())["checkpoint"] == "run-checkpoint"
    assert [path.name for path in tmp_path.iterdir()] == ["result.json"]


def test_secrets_for_a_remote_run_come_from_this_machine_and_skip_what_is_missing(monkeypatch, tmp_path):
    import polyplex_research.utils.phase as phase

    (tmp_path / ".netrc").write_text("machine api.wandb.ai\n  login user\n  password wandb-secret\n")
    (tmp_path / ".netrc").chmod(0o600)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    assert phase.remote_secrets() == {"WANDB_API_KEY": "wandb-secret"}


def test_the_package_archive_holds_the_sources_and_no_bytecode():
    import io
    import zipfile

    from polyplex_research.utils.phase import package_archive

    names = zipfile.ZipFile(io.BytesIO(package_archive())).namelist()
    assert "polyplex_research/utils/phase.py" in names
    assert not any(name.endswith(".pyc") for name in names)


def test_an_archive_runner_imports_the_shipped_package_and_sets_the_secrets(monkeypatch, tmp_path):
    import io
    import pickle
    import sys
    import zipfile

    import cloudpickle

    from polyplex_research.utils.phase import archive_runner

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("shipped_probe/__init__.py", "")
        archive.writestr(
            "shipped_probe/entry.py",
            "import os\ndef main(args):\n    return args, os.environ.get('WANDB_API_KEY'), __file__\n",
        )
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    # The runner travels by value, since the remote side cannot import this package.
    runner = pickle.loads(cloudpickle.dumps(archive_runner()))
    args, key, path = runner(buffer.getvalue(), "shipped_probe.entry", "main", ["--steps", "1"], {"WANDB_API_KEY": "k"})
    assert args == ["--steps", "1"] and key == "k"
    assert "letify-package-" in path
    sys.modules.pop("shipped_probe.entry", None)
    sys.modules.pop("shipped_probe", None)


def test_an_archive_runner_drops_the_paths_a_host_pythonpath_added(monkeypatch, tmp_path):
    import io
    import os
    import sys
    import zipfile

    from polyplex_research.utils.phase import archive_runner

    shadow = str(tmp_path / "other-site-packages")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("path_probe.py", "import os, sys\ndef main(args):\n    return list(sys.path), os.environ.get('PYTHONPATH')\n")
    monkeypatch.setenv("PYTHONPATH", shadow)
    monkeypatch.setattr(sys, "path", [shadow] + list(sys.path))
    path, variable = archive_runner()(buffer.getvalue(), "path_probe", "main", [], {})
    assert shadow not in path and variable is None
    sys.modules.pop("path_probe", None)


def test_the_extras_a_remote_run_installs_are_pinned_to_the_local_versions():
    import importlib.metadata

    from polyplex_research.utils.phase import locked_extras

    pins = locked_extras(("preprocess",))
    assert f"pyarrow=={importlib.metadata.version('pyarrow')}" in pins
    assert all("==" in pin and ">" not in pin for pin in pins)


def test_a_lost_session_is_retried_and_an_error_of_the_run_is_not():
    from letify.errors import ProtocolError, RemoteError

    from polyplex_research.utils.phase import retry_lost_sessions

    outcomes = [ProtocolError("lost"), ProtocolError("lost"), "done"]

    def call():
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert retry_lost_sessions(call, attempts=3) == "done"

    calls = []

    def failing():
        calls.append(1)
        raise RemoteError("ModuleNotFoundError: No module named 'x'")

    with pytest.raises(RemoteError):
        retry_lost_sessions(failing, attempts=3)
    assert len(calls) == 1


def test_a_retried_dense_run_resumes_from_its_stored_checkpoint(monkeypatch):
    import polyplex_research.utils.phase as phase

    stored = {"run": False}
    monkeypatch.setattr(phase, "stored_checkpoint_exists", lambda run_name: stored[run_name])
    args = ["--run-name", "run", "--max-steps", "10"]
    assert phase.resume_on_retry(args, "run") == args
    stored["run"] = True
    assert phase.resume_on_retry(args, "run") == args + ["--resume", "run"]
    with_resume = ["--run-name", "run", "--resume", "older"]
    assert phase.resume_on_retry(with_resume, "run") == with_resume


def test_a_stored_checkpoint_is_found_on_the_local_disk_the_drive_store_or_the_hub(monkeypatch, tmp_path):
    import letify

    import polyplex_research.utils.phase as phase

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(letify, "unstash", lambda name, path: None)
    monkeypatch.setattr(phase, "checkpoint_on_hub", lambda run_name: False)
    assert not phase.stored_checkpoint_exists("run")
    monkeypatch.setattr(phase, "checkpoint_on_hub", lambda run_name: run_name == "run")
    assert phase.stored_checkpoint_exists("run")
    monkeypatch.setattr(phase, "checkpoint_on_hub", lambda run_name: False)
    monkeypatch.setattr(letify, "unstash", lambda name, path: path if name == "run-checkpoint" else None)
    assert phase.stored_checkpoint_exists("run")
    monkeypatch.setattr(letify, "unstash", lambda name, path: None)
    (tmp_path / "runs" / "run" / "checkpoint").mkdir(parents=True)
    (tmp_path / "runs" / "run" / "checkpoint" / "manifest.json").write_text("{}")
    assert phase.stored_checkpoint_exists("run")
    (tmp_path / "runs" / "run" / "checkpoint" / "manifest.json").unlink()
    (tmp_path / "runs" / "run" / "checkpoint-250").mkdir()
    (tmp_path / "runs" / "run" / "checkpoint-250" / "manifest.json").write_text("{}")
    assert phase.stored_checkpoint_exists("run")


def test_the_dual_ffn_backbone_runs_every_projection_in_nvfp4(cpu_nvfp4):
    from polyplex_research.utils.assemble import quantize_frozen_backbone
    from polyplex_research.utils.quantization import FrozenNVFP4Linear

    from conftest import build_polyplex

    model = build_polyplex()
    quantize_frozen_backbone(model, "nvfp4")
    for layer in model.model.layers:
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert isinstance(getattr(layer.self_attn, projection), FrozenNVFP4Linear)
        for mlp in (layer.mlp.duplex_mlp, layer.mlp.language_mlp):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                assert isinstance(getattr(mlp, projection), FrozenNVFP4Linear)
    assert type(model.lm_head) is torch.nn.Linear


def test_a_dense_run_on_an_nvfp4_backbone_still_moves_the_normalizations(cpu_nvfp4):
    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.assemble import quantize_frozen_backbone
    from polyplex_research.utils.phase import dense_trainable_parameters, train_dense

    from conftest import build_polyplex

    model = build_polyplex()
    quantize_frozen_backbone(model, "nvfp4")
    trained = set(dense_trainable_parameters(model))
    before = {name: p.detach().clone() for name, p in model.named_parameters() if name in trained}
    train_dense(
        model,
        dense_collate_fn(SilentProcessor(), model),
        train_dataset=_dense_rows() * 2,
        eval_sets={},
        reference={},
        steps=2,
        batch_size=2,
        micro_batch_size=2,
        learning_rate=1e-2,
        warmup_steps=0,
        eval_every=100,
    )
    # Every layer's language_norm sits before a quantized FFN, so its gradient crosses NVFP4 projections.
    assert all(not torch.equal(p, before[name]) for name, p in model.named_parameters() if name in trained)


def test_the_eval_rows_are_the_first_rows_packed_from_the_held_out_documents(monkeypatch, tmp_path):
    import polyplex_research.utils.phase as phase

    class Pieces:
        def encode(self, text, add_special_tokens=False):
            return [len(word) for word in text.split()]

    asked = []

    def documents(language, split, tokens, tokenizer, max_files=1):
        asked.append((language, split, tokens))
        return ["a bb ccc " * 3, "dddd eeeee", "f gg"]

    monkeypatch.setattr(phase, "dense_documents", documents)
    rows = phase.dense_eval_rows("ko", Pieces(), str(tmp_path), eval_rows=2, row_frames=8, max_files=1)
    assert asked == [("ko", "valid", 16)]
    assert len(rows) == 2
    assert [segment.tolist() for segment in rows[0]["segments"]] == [[1, 2, 3, 1, 2, 3, 1]]


class StandInCorpus:
    texts = ["a  bb", "ccc dddd", "e", "ff"]
    made = []

    def __init__(self, language, split="train", filtered=False, max_files=None):
        StandInCorpus.made.append((language, split, filtered))
        self.request = ["repo", f"{language}-{filtered}"]

    def __iter__(self):
        return iter(self.texts)


def test_dense_documents_hand_back_the_pieces_they_counted(monkeypatch):
    import polyplex_research.utils.phase as phase

    Corpus = StandInCorpus

    class Pieces:
        calls = 0

        def encode(self, text, add_special_tokens=False):
            Pieces.calls += 1
            return [len(word) for word in text.split()]

    monkeypatch.setattr(phase, "MultilingualEduCorpus", Corpus)
    pieces = []
    documents = phase.dense_documents("ko", "train", 3, Pieces(), skip=1, pieces=pieces)
    assert documents == ["ccc dddd", "e"]
    assert pieces == [[3, 4], [1]] and Pieces.calls == 2
    assert phase.dense_documents("ko", "train", 3, Pieces(), skip=1) == documents


def test_dense_documents_are_read_back_from_the_document_cache(monkeypatch, tmp_path):
    import polyplex_research.utils.phase as phase

    reads = []

    class Corpus(StandInCorpus):
        def __iter__(self):
            reads.append(True)
            return super().__iter__()

    class Pieces:
        name_or_path = "pieces"

        def __len__(self):
            return 10

        def encode(self, text, add_special_tokens=False):
            return [len(word) for word in text.split()]

    monkeypatch.setattr(phase, "MultilingualEduCorpus", Corpus)
    monkeypatch.setenv("POLYPLEX_DOCUMENT_CACHE", str(tmp_path))
    first, second = [], []
    documents = phase.dense_documents("ko", "train", 3, Pieces(), skip=1, pieces=first)
    assert phase.dense_documents("ko", "train", 3, Pieces(), skip=1, pieces=second) == documents
    assert len(reads) == 1 and first == second == [[3, 4], [1]]
    phase.dense_documents("ko", "train", 3, Pieces(), skip=2)
    assert len(reads) == 2


def test_training_documents_can_come_from_the_filtered_source_leaving_out_what_was_trained_on(monkeypatch):
    import polyplex_research.utils.phase as phase
    from polyplex_research.datasets.webtext import text_key

    class Pieces:
        def encode(self, text, add_special_tokens=False):
            return [len(word) for word in text.split()]

    StandInCorpus.made.clear()
    monkeypatch.setattr(phase, "MultilingualEduCorpus", StandInCorpus)
    consumed = []
    documents = phase.dense_documents(
        "ja", "train", 100, Pieces(), filtered=True, exclude={text_key("ccc  dddd")}, consumed=consumed
    )
    assert documents == ["a  bb", "e", "ff"] and consumed == [4]
    assert StandInCorpus.made == [("ja", "train", True)]
    assert phase.trained_digests("ja", 2) == {text_key("a bb"), text_key("ccc dddd")}
    assert StandInCorpus.made[-1] == ("ja", "train", False)
    assert phase.trained_digests("ja", 0) == set()


def test_a_dense_run_trains_on_the_filtered_documents_unless_told_otherwise(monkeypatch):
    from polyplex_research.utils import phase

    called = {}
    monkeypatch.setattr(phase, "run_dense_adaptation", lambda **kwargs: called.update(kwargs) or {})
    phase.dense_main([])
    assert called["filtered_documents"] is True
    phase.dense_main(["--unfiltered-documents"])
    assert called["filtered_documents"] is False


def test_an_archive_runner_hands_path_arguments_on_as_strings():
    import io
    import pathlib
    import zipfile

    from polyplex_research.utils.phase import archive_runner

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("args_probe.py", "def main(args):\n    return args\n")
    assert archive_runner()(buffer.getvalue(), "args_probe", "main", ["--a", pathlib.Path("/x/y")], {}) == ["--a", "/x/y"]


# ---------------------------------------------------------------------------
# Dense check with the Polyplex vocabulary (SPEC S13.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "piece,byte_level,expected",
    [("<0xEC>", False, 1), ("▁안녕", False, 7), ("가", False, 3), ("Ġhello", True, 6), ("ìķĪ", True, 3)],
)
def test_a_piece_counts_the_utf8_bytes_of_the_text_it_stands_for(piece, byte_level, expected):
    from polyplex_research.utils.phase import piece_byte_length

    assert piece_byte_length(piece, byte_level) == expected


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) - ord("a") + 5 for character in text]


def test_bits_per_byte_divide_the_cross_entropy_of_the_scored_tokens_by_their_bytes():
    import math

    import torch.nn.functional as F

    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.phase import dense_bits_per_byte

    from conftest import build_polyplex

    model = build_polyplex()
    byte_lengths = torch.arange(model.config.vocab_size) % 3 + 1
    collate = dense_collate_fn(SilentProcessor(), model)
    texts = ["abcde", "fgh"]
    measured = dense_bits_per_byte(model, collate, CharacterTokenizer(), texts, byte_lengths)

    total, scored_bytes = 0.0, 0
    for text in texts:
        ids = CharacterTokenizer().encode(text)
        batch = collate([{"segments": [torch.tensor(ids)]}])
        labels = batch.pop("text_labels")
        logits = model(**batch, use_cache=False).logits[0, :-1]
        targets = torch.tensor(ids[1:])
        total += F.cross_entropy(logits[1:].float(), targets, reduction="sum").item()
        scored_bytes += int(byte_lengths[targets].sum())
    assert measured == pytest.approx(total / math.log(2) / scored_bytes, rel=1e-5)


def test_with_the_polyplex_vocabulary_the_dense_check_trains_new_rows_their_coefficients_and_the_norms():
    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.phase import dense_trainable_parameters, train_dense

    from conftest import VOCAB_SIZE, build_polyplex

    model = build_polyplex(text_composition_size=2, base_vocab_size=8)
    with torch.no_grad():
        model.text_composition.indices[40:VOCAB_SIZE] = torch.tensor([1, 2])
        model.text_composition.values[40:VOCAB_SIZE] = 0.5
    names = dense_trainable_parameters(model)
    assert {"lm_head.weight", "model.embed_tokens.weight", "text_composition.values"} <= set(names)
    assert "model.text_norm.weight" not in names
    assert "depth_decoder.model.text_embed_tokens.weight" not in names
    tables = {
        "head": model.lm_head.weight,
        "input": model.model.embed_tokens.weight,
        "coefficients": model.text_composition.values,
    }
    before = {name: table.detach().clone() for name, table in tables.items()}
    train_dense(
        model,
        dense_collate_fn(SilentProcessor(), model),
        train_dataset=_dense_rows() * 4,
        eval_sets={},
        reference={},
        steps=2,
        batch_size=2,
        micro_batch_size=2,
        learning_rate=1e-2,
        warmup_steps=0,
        eval_every=100,
    )
    for name, table in tables.items():
        # Base rows and the reserved row above the vocabulary stay as they were (SPEC S12.3a).
        torch.testing.assert_close(table[:8], before[name][:8], rtol=0, atol=0)
        torch.testing.assert_close(table[VOCAB_SIZE:], before[name][VOCAB_SIZE:], rtol=0, atol=0)
    assert not torch.equal(model.lm_head.weight[40:], before["head"][40:])
    assert not torch.equal(model.text_composition.values[40:VOCAB_SIZE], before["coefficients"][40:VOCAB_SIZE])


def test_unigram_bits_per_byte_score_every_token_but_the_first():
    import math

    from polyplex_research.utils.phase import unigram_bits_per_byte

    log_probs = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4] + [1e-9] * 60))
    byte_lengths = torch.tensor([1, 2, 1, 3] + [1] * 60)

    class Tokens:
        def encode(self, text, add_special_tokens=False):
            return [int(character) for character in text]

    measured = unigram_bits_per_byte(log_probs, Tokens(), ["0123", "31"], byte_lengths)
    nats = -(log_probs[1] + log_probs[2] + log_probs[3] + log_probs[1])
    assert measured == pytest.approx(nats.item() / math.log(2) / (2 + 1 + 3 + 2), rel=1e-6)


def test_the_dense_check_trains_every_parameter_of_an_appended_layer():
    from polyplex_research.utils.phase import dense_trainable_parameters

    from conftest import build_polyplex

    model = build_polyplex(text_composition_size=2, base_vocab_size=40, appended_layers=[1])
    names = dense_trainable_parameters(model)
    assert all(name in names for name, _ in model.model.layers[1].named_parameters(prefix="model.layers.1"))
    assert not any(name.startswith("model.layers.0.self_attn") for name in names)


def test_a_zero_language_scale_gives_the_model_with_the_language_ffn_off(text_batch):
    from polyplex_research.utils.phase import gate_language_ffn, language_ffn_off

    from conftest import build_polyplex

    model = build_polyplex()
    with torch.no_grad(), language_ffn_off(model):
        expected = model(input_ids=text_batch["input_ids"]).logits
    gate_language_ffn(model)
    assert model.config.language_scale
    with torch.no_grad():
        gated = model(input_ids=text_batch["input_ids"]).logits
    torch.testing.assert_close(gated, expected, rtol=0, atol=0)


def test_a_language_scale_multiplied_by_zero_gives_the_model_with_the_language_ffn_off(text_batch):
    from polyplex_research.utils.phase import gate_language_ffn, language_ffn_off, language_scale_multiplied

    from conftest import build_polyplex

    model = build_polyplex()
    with torch.no_grad(), language_ffn_off(model):
        expected = model(input_ids=text_batch["input_ids"]).logits
    gate_language_ffn(model)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.language_scale.fill_(0.5)
        scaled = model(input_ids=text_batch["input_ids"]).logits
        with language_scale_multiplied(model, 0.0):
            silenced = model(input_ids=text_batch["input_ids"]).logits
        with language_scale_multiplied(model, 2.0, layers=[1]):
            assert model.model.layers[0].mlp.language_scale[0].item() == 0.5
            assert model.model.layers[1].mlp.language_scale[0].item() == 1.0
    torch.testing.assert_close(silenced, expected, rtol=0, atol=0)
    assert not torch.equal(scaled, expected)
    assert all(layer.mlp.language_scale[0].item() == 0.5 for layer in model.model.layers)


def test_a_silenced_language_ffn_also_leaves_the_language_stream_unchanged(text_batch, tmp_path):
    from safetensors.torch import save_file

    from polyplex_research.utils.phase import language_ffn_off, language_scale_multiplied, open_language_stream

    from conftest import build_polyplex

    model = build_polyplex()
    torch.manual_seed(2)
    rotations = torch.stack([torch.linalg.qr(torch.randn(32, 32))[0] for _ in model.model.layers])
    path = tmp_path / "alignment.safetensors"
    save_file({"rotations": rotations, "scales": torch.tensor([0.5, 0.25])}, str(path))
    open_language_stream(model, str(path))
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.language_scale.fill_(0.5)
        with language_scale_multiplied(model, 0.0, layers=[0]):
            scaled = model(input_ids=text_batch["input_ids"]).logits
        with language_ffn_off(model, layers=[0]):
            silenced = model(input_ids=text_batch["input_ids"]).logits
        with language_scale_multiplied(model, 0.0):
            all_scaled = model(input_ids=text_batch["input_ids"]).logits
        with language_ffn_off(model):
            all_silenced = model(input_ids=text_batch["input_ids"]).logits
    # Layer 0's output still reaches layer 1 through the language stream when only its scale is 0.
    assert not torch.allclose(scaled, silenced)
    torch.testing.assert_close(all_scaled, all_silenced, rtol=1e-5, atol=1e-5)


def test_the_dense_mixture_follows_the_bytes_to_the_power_and_floors_english():
    import math

    from polyplex_research.utils.phase import dense_mixture

    sizes = {"en": 100.0, "kor_Hang": 1_000_000.0, "fra_Latn": 10_000.0, "tpi_Latn": 1.0}
    mixture = dense_mixture(sizes, temperature=0.3, english_floor=0.2)
    assert math.isclose(sum(mixture.values()), 1.0)
    assert math.isclose(mixture["en"], 0.2)
    # Every other language keeps its share of n ** 0.3, under the same normalization.
    assert math.isclose(mixture["ko"] / mixture["fra_Latn"], (1_000_000 / 10_000) ** 0.3)
    assert math.isclose(mixture["fra_Latn"] / mixture["tpi_Latn"], 10_000 ** 0.3)
    big = dense_mixture({"en": 1e12, "kor_Hang": 1.0}, temperature=0.3, english_floor=0.2)
    assert big["en"] > 0.2


def test_the_dense_mixture_keeps_only_the_chosen_languages(monkeypatch):
    import math

    import pytest

    import polyplex_research.utils.phase as phase

    sizes = {"en": 0.0, "kor_Hang": 1_000_000.0, "jpn_Jpan": 10_000.0, "fra_Latn": 100.0, "tpi_Latn": 1.0}
    mixture = phase.dense_mixture(sizes, languages=("en", "ko", "ja"))
    assert set(mixture) == {"en", "ko", "ja"}
    assert math.isclose(sum(mixture.values()), 1.0) and math.isclose(mixture["en"], 0.2)
    assert math.isclose(mixture["ko"] / mixture["ja"], 100**0.3)
    with pytest.raises(ValueError):
        phase.dense_mixture(sizes, languages=("en", "xx"))

    called = {}
    monkeypatch.setattr(phase, "run_dense_adaptation", lambda **kwargs: called.update(kwargs) or {"steps": 1})
    phase.dense_main(["--dense-languages", "en,ko,zh,ru,ar,hi,ja"])
    assert called["languages"] == ("en", "ko", "zh", "ru", "ar", "hi", "ja")
    phase.dense_main([])
    assert called["languages"] is None


def test_a_token_script_is_the_script_of_its_first_letter():
    from polyplex_research.utils.phase import token_script

    assert token_script(" 안녕") == "hangul"
    assert token_script("漢字") == "han"
    assert token_script("ひら") == "hiragana" and token_script("カタ") == "katakana"
    assert token_script("▁word") == "latin" and token_script("слово") == "cyrillic"
    assert token_script("كلمة") == "arabic" and token_script("शब्द") == "devanagari"
    assert token_script(" 12") == "digit" and token_script(" .,") == "other"


def test_token_losses_are_summed_into_bits_per_byte_by_group():
    import math

    from polyplex_research.utils.phase import grouped_bits_per_byte

    losses = [
        {"loss": math.log(2) * 4, "bytes": 2, "script": "han", "new": True, "position": 3},
        {"loss": math.log(2) * 2, "bytes": 2, "script": "han", "new": False, "position": 300},
        {"loss": math.log(2) * 3, "bytes": 3, "script": "latin", "new": False, "position": 5},
    ]
    groups = grouped_bits_per_byte(losses, positions=(256,))
    assert groups["script"]["han"] == {"bits_per_byte": 1.5, "byte_share": 4 / 7}
    assert groups["script"]["latin"] == {"bits_per_byte": 1.0, "byte_share": 3 / 7}
    assert groups["vocabulary"]["new"]["bits_per_byte"] == 2.0
    assert groups["position"]["<256"]["bits_per_byte"] == 1.4
    assert groups["position"][">=256"]["bits_per_byte"] == 1.0


def test_new_token_losses_are_grouped_by_how_often_training_saw_the_token():
    import math

    from polyplex_research.utils.phase import grouped_bits_per_byte

    losses = [
        {"loss": math.log(2) * 4, "bytes": 2, "script": "han", "new": True, "position": 3, "id": 10},
        {"loss": math.log(2) * 2, "bytes": 2, "script": "han", "new": True, "position": 4, "id": 11},
        {"loss": math.log(2) * 1, "bytes": 1, "script": "han", "new": True, "position": 5, "id": 12},
        {"loss": math.log(2) * 3, "bytes": 3, "script": "latin", "new": False, "position": 6, "id": 1},
    ]
    counts = [0] * 10 + [0, 5, 500]
    groups = grouped_bits_per_byte(losses, positions=(256,), counts=counts)
    assert groups["exposure"]["new 0"] == {"bits_per_byte": 2.0, "byte_share": 2 / 8}
    assert groups["exposure"]["new 1-9"]["bits_per_byte"] == 1.0
    assert groups["exposure"]["new >=100"]["bits_per_byte"] == 1.0
    assert groups["exposure"]["base"]["byte_share"] == 3 / 8
    assert "exposure" not in grouped_bits_per_byte(losses, positions=(256,))


def test_the_language_scale_trains_and_gets_a_gradient_at_zero(text_batch):
    from polyplex_research.utils.phase import dense_trainable_parameters, gate_language_ffn

    from conftest import build_polyplex

    model = build_polyplex()
    gate_language_ffn(model)
    names = dense_trainable_parameters(model)
    assert {"model.layers.0.mlp.language_scale", "model.layers.1.mlp.language_scale"} <= set(names)
    model(input_ids=text_batch["input_ids"], text_labels=text_batch["input_ids"]).loss.backward()
    assert model.model.layers[0].mlp.language_scale.grad.abs().sum() > 0


def test_a_language_lora_starts_as_the_model_without_it_and_trains_alone(text_batch):
    from polyplex_research.utils.phase import dense_trainable_parameters, language_ffn_off, open_language_lora

    from conftest import build_polyplex

    model = build_polyplex()
    with torch.no_grad():
        before = model(input_ids=text_batch["input_ids"]).logits
    open_language_lora(model, 4)
    assert model.config.language_lora_rank == 4
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=text_batch["input_ids"]).logits, before, rtol=0, atol=0)
    names = set(dense_trainable_parameters(model))
    assert "model.layers.0.mlp.language_mlp.down_lora.1.weight" in names
    assert "model.layers.0.mlp.language_mlp.down_proj.weight" not in names
    model(input_ids=text_batch["input_ids"], text_labels=text_batch["input_ids"]).loss.backward()
    assert model.model.layers[0].mlp.language_mlp.up_lora[1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.model.layers[0].mlp.language_mlp.down_lora[1].weight.fill_(0.1)
        changed = model(input_ids=text_batch["input_ids"]).logits
        assert not torch.equal(changed, before)
        with language_ffn_off(model):
            silent = model(input_ids=text_batch["input_ids"]).logits
        model.model.layers[0].mlp.language_mlp.down_lora[1].weight.zero_()
        with language_ffn_off(model):
            torch.testing.assert_close(model(input_ids=text_batch["input_ids"]).logits, silent, rtol=0, atol=0)


def test_an_opened_language_stream_starts_as_the_model_with_the_language_ffn_off(text_batch, tmp_path):
    from safetensors.torch import save_file

    from polyplex_research.utils.phase import language_ffn_off, open_language_stream

    from conftest import build_polyplex

    model = build_polyplex()
    with torch.no_grad(), language_ffn_off(model):
        expected = model(input_ids=text_batch["input_ids"]).logits
    torch.manual_seed(2)
    rotations = torch.stack([torch.linalg.qr(torch.randn(32, 32))[0] for _ in model.model.layers])
    path = tmp_path / "alignment.safetensors"
    save_file({"rotations": rotations, "scales": torch.tensor([0.5, 0.25])}, str(path))
    open_language_stream(model, str(path))
    assert model.config.language_stream and model.config.language_scale
    torch.testing.assert_close(model.model.layers[1].mlp.stream_rotation, rotations[1])
    assert model.model.layers[1].mlp.stream_scale.item() == 0.25
    with torch.no_grad():
        streamed = model(input_ids=text_batch["input_ids"]).logits
    torch.testing.assert_close(streamed, expected, rtol=1e-5, atol=1e-5)


def test_bits_per_byte_with_the_language_ffn_off_silence_it_and_put_it_back():
    import copy

    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.phase import dense_bits_per_byte, language_ffn_off_bits_per_byte

    from conftest import build_polyplex

    model = build_polyplex()
    byte_lengths = torch.arange(model.config.vocab_size) % 3 + 1
    collate = dense_collate_fn(SilentProcessor(), model)
    texts = ["abcde", "fgh"]
    weights = [layer.mlp.language_mlp.down_proj.weight.detach().clone() for layer in model.model.layers]
    measured = language_ffn_off_bits_per_byte(model, collate, CharacterTokenizer(), texts, byte_lengths)

    silent = copy.deepcopy(model)
    with torch.no_grad():
        for layer in silent.model.layers:
            layer.mlp.language_mlp.down_proj.weight.zero_()
    assert measured == pytest.approx(dense_bits_per_byte(silent, collate, CharacterTokenizer(), texts, byte_lengths))
    for layer, weight in zip(model.model.layers, weights):
        assert torch.equal(layer.mlp.language_mlp.down_proj.weight, weight)


def test_the_new_token_share_is_the_mean_probability_of_the_ids_past_the_base_vocabulary():
    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.phase import new_token_share

    from conftest import build_polyplex

    model = build_polyplex()
    collate = dense_collate_fn(SilentProcessor(), model)
    texts = ["abcde", "fgh"]
    measured = new_token_share(model, collate, CharacterTokenizer(), texts, base=40)

    shares = []
    for text in texts:
        batch = collate([{"segments": [torch.tensor(CharacterTokenizer().encode(text))]}])
        batch.pop("text_labels")
        logits = model(**batch, use_cache=False).logits[0, 1:-1]
        shares.append(logits.float().softmax(-1)[:, 40:].sum(-1))
    assert measured == pytest.approx(torch.cat(shares).mean().item(), rel=1e-5)


def test_the_mean_probabilities_average_the_softmax_over_the_scored_positions():
    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.phase import mean_probabilities, new_token_share

    from conftest import build_polyplex

    model = build_polyplex()
    collate = dense_collate_fn(SilentProcessor(), model)
    texts = ["abcde", "fgh"]
    probabilities = mean_probabilities(model, collate, CharacterTokenizer(), texts)
    assert probabilities.shape == (model.config.vocab_size,)
    assert probabilities.sum().item() == pytest.approx(1.0, rel=1e-5)
    assert probabilities[40:].sum().item() == pytest.approx(
        new_token_share(model, collate, CharacterTokenizer(), texts, base=40), rel=1e-5
    )


# ---------------------------------------------------------------------------
# Dense run schedule, optimizer and checkpoints (SPEC S13.4, S7.2)
# ---------------------------------------------------------------------------


def test_the_wsd_schedule_warms_up_holds_and_decays_only_at_the_end():
    import math

    from polyplex_research.utils.phase import wsd_factor

    assert [wsd_factor(step, 4, 100, 0) for step in (1, 2, 4, 50, 100)] == [0.25, 0.5, 1.0, 1.0, 1.0]
    assert wsd_factor(80, 4, 100, 20) == 1.0
    assert wsd_factor(95, 4, 100, 20) == pytest.approx(0.5 * (1 + math.cos(math.pi * 15 / 20)))
    assert wsd_factor(100, 4, 100, 20) == pytest.approx(0.0)


def test_the_dense_optimizer_steps_matrices_with_muon_and_tables_norms_and_coefficients_with_adamw():
    from polyplex_research.utils.phase import dense_optimizer

    masters = {
        "model.layers.1.self_attn.q_proj.weight": torch.zeros(8, 8, requires_grad=True),
        "lm_head.weight": torch.zeros(10, 8, requires_grad=True),
        "model.embed_tokens.weight": torch.zeros(11, 8, requires_grad=True),
        "text_composition.values": torch.zeros(11, 2, requires_grad=True),
        "model.layers.0.mlp.language_norm.weight": torch.zeros(8, requires_grad=True),
    }
    optimizer = dense_optimizer(masters, 1e-3)
    muon = {id(parameter) for group in optimizer.muon.param_groups for parameter in group["params"]}
    assert muon == {id(masters["model.layers.1.self_attn.q_proj.weight"])}
    tables = {id(parameter) for group in optimizer.tables.param_groups for parameter in group["params"]}
    assert tables == {id(masters["lm_head.weight"]), id(masters["model.embed_tokens.weight"])}
    adamw = [parameter for group in optimizer.adamw.param_groups for parameter in group["params"]]
    assert {id(parameter) for parameter in adamw} == {id(tensor) for tensor in masters.values()} - muon - tables
    assert all(group["weight_decay"] == 0.0 for group in optimizer.adamw.param_groups)


def test_an_adamw_state_of_the_tables_carries_over_to_the_factored_tables():
    from polyplex_research.utils.optimization import MuonWithAdamW
    from polyplex_research.utils.phase import dense_optimizer, upgrade_dense_optimizer_state

    torch.manual_seed(0)
    masters = {
        "model.layers.1.self_attn.q_proj.weight": torch.randn(8, 8, requires_grad=True),
        "lm_head.weight": torch.randn(10, 8, requires_grad=True),
        "text_composition.values": torch.randn(11, 2, requires_grad=True),
        "model.embed_tokens.weight": torch.randn(11, 8, requires_grad=True),
        "model.layers.0.mlp.language_norm.weight": torch.randn(8, requires_grad=True),
    }
    # The layout before S7.2a: every table on AdamW, in the order of the masters.
    rest = [tensor for name, tensor in masters.items() if name != "model.layers.1.self_attn.q_proj.weight"]
    old = MuonWithAdamW([masters["model.layers.1.self_attn.q_proj.weight"]], [{"params": rest, "weight_decay": 0.0}], lr=1e-3)
    for tensor in masters.values():
        tensor.grad = torch.randn_like(tensor)
    old.step()
    saved = old.state_dict()

    optimizer = dense_optimizer(masters, 1e-3)
    optimizer.load_state_dict(upgrade_dense_optimizer_state(saved, masters, optimizer))
    head = optimizer.tables.state[masters["lm_head.weight"]]
    previous = old.adamw.state[masters["lm_head.weight"]]
    assert torch.equal(head["exp_avg"], previous["exp_avg"])
    torch.testing.assert_close(head["exp_avg_sq_row"], previous["exp_avg_sq"].mean(1))
    values = optimizer.adamw.state[masters["text_composition.values"]]
    assert torch.equal(values["exp_avg_sq"], old.adamw.state[masters["text_composition.values"]]["exp_avg_sq"])
    # A state already in the new layout is passed through.
    assert upgrade_dense_optimizer_state(optimizer.state_dict(), masters, optimizer) == optimizer.state_dict()


def test_a_sparse_gradient_keeps_only_the_trained_rows():
    from polyplex_research.utils.phase import keep_trained_rows

    gradient = torch.sparse_coo_tensor(torch.tensor([[1, 4, 6, 9]]), torch.ones(4, 2), (10, 2))
    kept = keep_trained_rows(gradient, 3, 8)
    assert kept.is_sparse and sorted(kept.coalesce().indices()[0].tolist()) == [4, 6]
    dense = torch.ones(10, 2)
    assert keep_trained_rows(dense, 3, 8).sum(1).tolist() == [0, 0, 0, 2, 2, 2, 2, 2, 0, 0]


def _trained_state(seed=0):
    from polyplex_research.utils.phase import dense_optimizer

    torch.manual_seed(seed)
    trained = {"a.weight": torch.randn(8, 8).to(torch.bfloat16), "b.weight": torch.randn(8).to(torch.bfloat16)}
    masters = {name: tensor.float().clone().requires_grad_(True) for name, tensor in trained.items()}
    optimizer = dense_optimizer(masters, 1e-2)
    for master in masters.values():
        master.grad = torch.randn_like(master)
    optimizer.step()
    return trained, masters, optimizer


def test_a_checkpoint_keeps_every_tensor_in_its_own_dtype_and_hands_over_one_file_at_a_time(tmp_path):
    from polyplex_research.utils.phase import dense_optimizer, read_checkpoint, write_checkpoint

    trained, masters, optimizer = _trained_state()
    handed = []

    def hand_over(path):
        handed.append((path.name, sorted(item.name for item in tmp_path.iterdir())))
        path.unlink()

    write_checkpoint(tmp_path, 7, trained, masters, optimizer, {"documents_used": {"en": 3}}, hand_over)
    assert handed[-1][0] == "manifest.json"
    # Each file leaves before the next is written, so at most one sits on the disk.
    assert all(len(present) == 1 for _, present in handed)
    assert not list(tmp_path.iterdir())


def test_a_checkpoint_is_one_tensor_file_and_a_manifest(tmp_path):
    import polyplex_research.utils.phase as phase

    trained, masters, optimizer = _trained_state()
    names = phase.write_checkpoint(tmp_path, 7, trained, masters, optimizer, {})
    assert names == ["checkpoint.safetensors", "manifest.json"]
    assert sorted(item.name for item in tmp_path.iterdir()) == names


def test_a_checkpoint_reads_back_what_was_written(tmp_path):
    from polyplex_research.utils.phase import dense_optimizer, read_checkpoint, write_checkpoint

    trained, masters, optimizer = _trained_state()
    write_checkpoint(tmp_path, 7, trained, masters, optimizer, {"documents_used": {"en": 3}})
    state = read_checkpoint(tmp_path)
    assert state["step"] == 7 and state["extra"] == {"documents_used": {"en": 3}}
    assert read_checkpoint(tmp_path, tensors=False) == {"step": 7, "extra": {"documents_used": {"en": 3}}}
    for name, tensor in trained.items():
        assert state["weights"][name].dtype == torch.bfloat16 and torch.equal(state["weights"][name], tensor)
        assert state["masters"][name].dtype == torch.float32 and torch.equal(state["masters"][name], masters[name])
    restored = {name: tensor.clone().requires_grad_(True) for name, tensor in state["masters"].items()}
    fresh = dense_optimizer(restored, 1e-2)
    fresh.load_state_dict(state["optimizer"])
    saved = optimizer.state_dict()
    for side in ("muon", "adamw"):
        for index, entries in saved[side]["state"].items():
            for key, value in entries.items():
                assert torch.equal(fresh.state_dict()[side]["state"][index][key], value), (side, key)


def _dense_run(model, steps, **kwargs):
    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.phase import train_dense

    return train_dense(
        model,
        dense_collate_fn(SilentProcessor(), model),
        train_dataset=_dense_rows() * 8,
        eval_sets={},
        reference={},
        steps=steps,
        batch_size=2,
        micro_batch_size=2,
        learning_rate=1e-2,
        warmup_steps=1,
        eval_every=100,
        **kwargs,
    )


def test_a_dense_run_moves_to_the_shared_stream_rotation_once_after_the_switch_step():
    from conftest import build_polyplex

    from polyplex_research.utils.assemble import fold_language_stream

    model = build_polyplex(language_scale=True, language_stream=True)
    events = []

    def switch(switched):
        events.append("switch")
        fold_language_stream(switched, torch.eye(32), 0.5)

    _dense_run(model, 4, stream_switch=switch, switch_step=2, log=lambda metrics: events.append(metrics["step"]))
    assert model.config.language_stream_folded
    assert [event for event in events if event == "switch" or event in (2, 3)] == [2, "switch", 3]
    assert events.count("switch") == 1


def test_a_run_resumed_past_the_switch_step_switches_before_its_first_step():
    from conftest import build_polyplex

    from polyplex_research.utils.assemble import fold_language_stream

    model = build_polyplex(language_scale=True, language_stream=True)
    switched = []
    _dense_run(model, 2, stream_switch=lambda m: switched.append(1) or fold_language_stream(m, torch.eye(32), 1.0),
               switch_step=0)
    assert switched == [1] and model.config.language_stream_folded


def test_a_checkpoint_keeps_the_buffers_it_is_given(tmp_path):
    from polyplex_research.utils.phase import read_checkpoint, write_checkpoint

    trained, masters, optimizer = _trained_state()
    gains = torch.rand(3, 4)
    write_checkpoint(tmp_path, 7, trained, masters, optimizer, {}, buffers={"stream_gains": gains})
    assert torch.equal(read_checkpoint(tmp_path)["buffers"]["stream_gains"], gains)


def test_a_dense_run_writes_a_checkpoint_every_few_steps_and_at_its_end():
    from conftest import build_polyplex

    saved = []
    _dense_run(build_polyplex(), 5, save_every=2, checkpoint=lambda step, *state: saved.append(step))
    assert saved == [2, 4, 5]


def test_a_dense_run_saves_the_last_finished_step_when_it_fails():
    from conftest import build_polyplex

    model = build_polyplex()
    calls = {"count": 0}
    forward = model.forward

    def failing(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("out of memory")
        return forward(*args, **kwargs)

    model.forward = failing
    saved = []
    with pytest.raises(RuntimeError, match="out of memory"):
        _dense_run(model, 5, save_every=100, checkpoint=lambda step, *state: saved.append(step))
    assert saved == [2]


def test_a_dense_run_saves_and_stops_when_asked_to():
    from conftest import build_polyplex

    saved = []
    result = _dense_run(
        build_polyplex(), 5, save_every=100, checkpoint=lambda step, *state: saved.append(step), stop=lambda: True
    )
    assert saved == [1] and result["steps"] == 1


def test_a_resumed_dense_run_starts_from_the_saved_step_masters_and_optimizer():
    from conftest import build_polyplex

    first = build_polyplex()
    states = []
    _dense_run(first, 2, save_every=100, checkpoint=lambda step, trained, masters, optimizer: states.append(
        (step, {name: tensor.detach().clone() for name, tensor in masters.items()}, optimizer.state_dict())
    ))
    step, masters, optimizer_state = states[-1]
    second = build_polyplex()
    logged = []
    weights = {name: parameter.detach().clone() for name, parameter in first.named_parameters() if parameter.requires_grad}
    result = _dense_run(
        second,
        1,
        resume_state={"step": step, "weights": weights, "masters": masters, "optimizer": optimizer_state},
        log=logged.append,
    )
    assert result["steps"] == 3
    assert [entry["step"] for entry in logged if "train/loss" in entry] == [3]
    # The checkpoint tensors are let go once loaded, so they do not sit beside the model through training.
    resumed = {"step": step, "weights": weights, "masters": masters, "optimizer": optimizer_state}
    _dense_run(build_polyplex(), 1, resume_state=resumed)
    assert resumed == {"step": step}
    # No evaluation before the first step of a resumed run.
    assert result["evaluations"][0]["step"] == 3
    resumed = _dense_run(
        build_polyplex(),
        1,
        resume_state={"step": step, "weights": weights, "masters": masters, "optimizer": optimizer_state},
        evaluate_at_start=True,
    )
    assert [entry["step"] for entry in resumed["evaluations"]] == [2, 3]



def test_model_downloads_are_freed_only_on_a_disk_the_run_owns(monkeypatch):
    import polyplex_research.utils.phase as phase

    freed = []
    monkeypatch.setattr(phase, "_delete_cached_repos", lambda repo_ids: freed.append(list(repo_ids)))
    monkeypatch.delenv("POLYPLEX_EPHEMERAL_DISK", raising=False)
    phase.free_downloads(["a/b"])
    assert freed == []
    monkeypatch.setenv("POLYPLEX_EPHEMERAL_DISK", "1")
    phase.free_downloads(["a/b", "c/d"])
    assert freed == [["a/b", "c/d"]]


def test_only_rented_machines_get_an_ephemeral_disk():
    from polyplex_research.utils.phase import ephemeral_disk

    assert ephemeral_disk("colab_pro_plus.RTX_PRO_6000") and ephemeral_disk("kaggle_30")
    assert not ephemeral_disk("local") and not ephemeral_disk("dept_gpu.A6000")


def test_a_dense_run_keeps_float32_masters_only_for_vectors_and_coefficients_and_steps_the_rest_in_bfloat16():
    from conftest import build_polyplex

    model = build_polyplex(text_composition_size=2, base_vocab_size=40, appended_layers=[1]).to(torch.bfloat16)
    seen = {}

    def keep(step, trained, masters, optimizer):
        seen.update(masters=dict(masters), trained=dict(trained), state=optimizer.state_dict())

    _dense_run(model, 1, save_every=100, checkpoint=keep)
    trained, masters = seen["trained"], seen["masters"]
    expected = {name for name, parameter in trained.items() if parameter.ndim == 1 or name == "text_composition.values"}
    assert set(masters) == expected and all(master.dtype == torch.float32 for master in masters.values())
    assert "lm_head.weight" not in masters and "model.layers.1.self_attn.q_proj.weight" not in masters
    for side in ("muon", "adamw"):
        for entries in seen["state"][side]["state"].values():
            for key, value in entries.items():
                if torch.is_tensor(value) and value.ndim == 2 and value.shape[-1] > 2:
                    assert value.dtype == torch.bfloat16, (side, key)


def test_the_depth_decoder_waits_on_the_cpu_while_text_trains():
    from polyplex_research.utils.phase import depth_decoder_on

    from conftest import build_polyplex

    model = build_polyplex()
    with depth_decoder_on(model, "cpu"):
        assert next(model.depth_decoder.parameters()).device.type == "cpu"
    assert next(model.depth_decoder.parameters()).device == next(model.model.parameters()).device


def test_the_language_ffn_is_silenced_without_touching_an_nvfp4_projection(cpu_nvfp4):
    from polyplex_research.utils.assemble import quantize_frozen_backbone
    from polyplex_research.utils.phase import language_ffn_off

    from conftest import build_polyplex

    model = build_polyplex()
    quantize_frozen_backbone(model, "nvfp4")
    hidden = torch.randn(1, 3, 32)
    mlp = model.model.layers[0].mlp
    with language_ffn_off(model):
        silent = mlp(hidden)
    expected = mlp.duplex_mlp(mlp.duplex_norm(hidden))
    torch.testing.assert_close(silent, expected)
    assert not torch.equal(mlp(hidden), silent)


def test_the_language_ffn_is_silenced_in_a_compiled_layer_too():
    from polyplex_research.utils.phase import language_ffn_off

    from conftest import build_polyplex

    torch._dynamo.reset()
    model = build_polyplex()
    layer = model.model.layers[0]
    layer.mlp.compile(backend="eager")
    hidden = torch.randn(1, 3, 32)
    mlp = layer.mlp
    both = mlp(hidden)
    with language_ffn_off(model):
        silent = mlp(hidden)
    expected = mlp.duplex_mlp(mlp.duplex_norm(hidden))
    torch.testing.assert_close(silent, expected)
    torch.testing.assert_close(mlp(hidden), both)


def test_bits_per_byte_over_the_first_ids_renormalizes_over_them_alone():
    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.phase import dense_bits_per_byte

    from conftest import build_polyplex

    model = build_polyplex()
    collate = dense_collate_fn(SilentProcessor(), model)
    byte_lengths = torch.ones(model.config.vocab_size + 1, dtype=torch.long)
    texts = ["abcdefg"]
    full = dense_bits_per_byte(model, collate, CharacterTokenizer(), texts, byte_lengths)
    limited = dense_bits_per_byte(model, collate, CharacterTokenizer(), texts, byte_lengths, vocabulary=20)
    assert limited < full
    # "x", "y" and "z" are ids 28 to 30: past the first 20 ids, so nothing is scored.
    assert dense_bits_per_byte(model, collate, CharacterTokenizer(), ["axyz"], byte_lengths, vocabulary=20) == 0


def test_the_language_ffn_output_can_be_scaled_and_put_back():
    from polyplex_research.utils.phase import language_ffn_scaled

    from conftest import build_polyplex

    model = build_polyplex()
    mlp = model.model.layers[0].mlp
    hidden = torch.randn(1, 3, 32)
    both = mlp(hidden)
    duplex = mlp.duplex_mlp(mlp.duplex_norm(hidden))
    language = mlp.language_mlp(mlp.language_norm(hidden))
    with language_ffn_scaled(model, 0.25):
        torch.testing.assert_close(mlp(hidden), duplex + 0.25 * language)
    torch.testing.assert_close(mlp(hidden), both)


def test_expandable_segments_are_asked_for_only_when_a_gpu_is_there(monkeypatch):
    import polyplex_research.utils.phase as phase

    asked = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", lambda settings: asked.append(settings))
    phase.use_expandable_segments()
    assert asked == []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    phase.use_expandable_segments()
    assert asked == ["expandable_segments:True"]


def test_checkpoints_upload_on_their_own_thread_only_from_an_ephemeral_disk(monkeypatch, tmp_path):
    import threading

    import polyplex_research.utils.phase as phase

    release, stashed = threading.Event(), []

    def stash(run_name, path):
        release.wait(5)
        stashed.append(path.name)
        path.unlink()

    monkeypatch.setattr(phase, "stash_checkpoint_file", stash)
    monkeypatch.delenv("POLYPLEX_EPHEMERAL_DISK", raising=False)
    assert phase.checkpoint_hand_over("run") is None
    monkeypatch.setenv("POLYPLEX_EPHEMERAL_DISK", "1")
    uploader = phase.checkpoint_hand_over("run")
    trained, masters, optimizer = _trained_state()
    for step in (250, 500):
        directory = tmp_path / f"checkpoint-{step}"
        phase.write_checkpoint(directory, step, trained, masters, optimizer, {})
        uploader.submit(directory)
    # The run is not held up: both checkpoints are on the disk while the first upload still waits.
    assert stashed == [] and (tmp_path / "checkpoint-500" / "manifest.json").exists()
    release.set()
    uploader.close()
    assert stashed == ["checkpoint.safetensors", "manifest.json"] * 2
    assert not list(tmp_path.iterdir())


def test_closing_the_uploader_waits_a_bounded_time_and_keeps_unfinished_files(monkeypatch, tmp_path, caplog):
    import threading

    import polyplex_research.utils.phase as phase

    release = threading.Event()
    monkeypatch.setattr(phase, "stash_checkpoint_file", lambda run_name, path: release.wait(5))
    uploader = phase.CheckpointUploader("run")
    trained, masters, optimizer = _trained_state()
    directory = tmp_path / "checkpoint-250"
    phase.write_checkpoint(directory, 250, trained, masters, optimizer, {})
    uploader.submit(directory)
    with caplog.at_level("WARNING", logger=phase.logger.name):
        assert uploader.close(timeout=0.1) is False
    assert "still uploading" in caplog.text
    assert (directory / "manifest.json").exists() and (directory / "checkpoint.safetensors").exists()
    release.set()


def test_a_resume_reads_the_local_checkpoint_before_the_drive_store(monkeypatch, tmp_path):
    import polyplex_research.utils.phase as phase

    fetched = []
    monkeypatch.setattr(phase, "unstash_checkpoint", lambda run_name, directory: fetched.append(run_name) or directory)
    runs = tmp_path / "runs"
    assert str(phase.resume_checkpoint("old", str(runs / "new"))) == str(runs / "new" / "resume")
    assert fetched == ["old"]
    (runs / "old" / "checkpoint").mkdir(parents=True)
    (runs / "old" / "checkpoint" / "manifest.json").write_text("{}")
    assert phase.resume_checkpoint("old", str(runs / "new")) == runs / "old" / "checkpoint"
    assert fetched == ["old"]


def test_a_resume_reads_the_latest_local_checkpoint_the_uploader_left(monkeypatch, tmp_path):
    import polyplex_research.utils.phase as phase

    monkeypatch.setattr(phase, "unstash_checkpoint", lambda run_name, directory: pytest.fail("the store is asked"))
    runs = tmp_path / "runs"
    for step in (1750, 2000):
        (runs / "old" / f"checkpoint-{step}").mkdir(parents=True)
        (runs / "old" / f"checkpoint-{step}" / "manifest.json").write_text(json.dumps({"step": step}))
    assert phase.resume_checkpoint("old", str(runs / "new")) == runs / "old" / "checkpoint-2000"


def test_a_dense_run_with_the_composition_steps_the_input_table_sparsely_and_keeps_its_base_rows():
    from conftest import build_polyplex

    model = build_polyplex(text_composition_size=2, base_vocab_size=8)
    table = model.get_input_embeddings()
    head = model.lm_head.weight.detach().clone()
    rows = table.weight.detach().clone()
    result = _dense_run(model, 2)
    assert result["steps"] == 2
    assert torch.equal(table.weight.detach()[:8], rows[:8]) and torch.equal(model.lm_head.weight.detach()[:8], head[:8])
    assert not torch.equal(table.weight.detach()[8:15], rows[8:15])
    # Rows the batches never read stay where they were.
    assert torch.equal(table.weight.detach()[15:64], rows[15:64])
    assert table.sparse is False


def test_a_checkpoint_folds_the_residual_alignment_into_its_language_side(tmp_path):
    from safetensors.torch import load_file, save_file

    from polyplex_research.utils.phase import dense_optimizer, fold_checkpoint_alignment, read_checkpoint, write_checkpoint

    torch.manual_seed(0)
    gains = [torch.rand(8) + 0.5, torch.rand(8) + 0.5]
    trained = {
        "model.layers.0.mlp.language_norm.weight": gains[0].to(torch.bfloat16),
        "model.layers.1.mlp.language_norm.weight": gains[1].to(torch.bfloat16),
        "model.layers.1.mlp.language_mlp.gate_proj.weight": torch.randn(12, 8).to(torch.bfloat16),
        "model.layers.1.mlp.language_mlp.up_proj.weight": torch.randn(12, 8).to(torch.bfloat16),
        "model.layers.1.mlp.language_mlp.down_proj.weight": torch.randn(8, 12).to(torch.bfloat16),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(8, 8).to(torch.bfloat16),
        "lm_head.weight": torch.randn(10, 8).to(torch.bfloat16),
        "text_composition.values": torch.randn(11, 2).to(torch.bfloat16),
    }
    masters = {name: tensor.float().clone().requires_grad_(True) for name, tensor in trained.items() if name.endswith(
        ("language_norm.weight", "text_composition.values"))}
    masters["model.layers.0.mlp.language_norm.weight"].data.copy_(gains[0])
    masters["model.layers.1.mlp.language_norm.weight"].data.copy_(gains[1])
    stepped = {name: masters.get(name, tensor.requires_grad_(True)) for name, tensor in trained.items()}
    optimizer = dense_optimizer(stepped, 1e-3)
    for tensor in stepped.values():
        tensor.grad = torch.randn_like(tensor)
    optimizer.step()
    source = tmp_path / "runs" / "old" / "checkpoint"
    write_checkpoint(source, 9, trained, masters, optimizer, {"documents_used": {"en": 4}})
    alignment = tmp_path / "alignment.safetensors"
    save_file({"rotations": torch.eye(8).expand(2, 8, 8).contiguous(), "scales": torch.tensor([1.0, 2.0])}, str(alignment))

    folded = fold_checkpoint_alignment(str(source), str(tmp_path / "runs" / "new"), str(alignment))
    assert sorted(item.name for item in (tmp_path / "runs" / "new").glob("checkpoint*")) == ["checkpoint-9"]
    state = read_checkpoint(tmp_path / "runs" / "new" / "checkpoint-9")
    assert state["step"] == 9 and state["extra"]["documents_used"] == {"en": 4}
    for layer in (0, 1):
        name = f"model.layers.{layer}.mlp.language_norm.weight"
        assert torch.equal(state["weights"][name], torch.ones(8, dtype=torch.bfloat16))
        assert torch.equal(state["masters"][name], torch.ones(8))
    gate = trained["model.layers.1.mlp.language_mlp.gate_proj.weight"].float() * masters[
        "model.layers.1.mlp.language_norm.weight"].detach()
    torch.testing.assert_close(state["weights"]["model.layers.1.mlp.language_mlp.gate_proj.weight"].float(), gate,
                               atol=0.05, rtol=0.02)
    down = 2 * trained["model.layers.1.mlp.language_mlp.down_proj.weight"].float()
    assert torch.equal(state["weights"]["model.layers.1.mlp.language_mlp.down_proj.weight"].float(), down)
    # The folded parameters start their optimizer state afresh; the others keep theirs.
    fresh = dense_optimizer({name: tensor.detach().clone().requires_grad_(True) for name, tensor in stepped.items()}, 1e-3)
    fresh.load_state_dict(state["optimizer"])
    assert len(fresh.muon.state) == 1 and len(fresh.adamw.state) == 1 and len(fresh.tables.state) == 1
    # The gains folded are the trained ones, read from the float32 masters.
    stepped_gains = [masters[f"model.layers.{layer}.mlp.language_norm.weight"].detach() for layer in (0, 1)]
    torch.testing.assert_close(load_file(folded)["gains"], torch.stack(stepped_gains), atol=0, rtol=0)


def test_a_local_checkpoint_replaces_the_previous_one_only_once_it_is_whole(tmp_path):
    from polyplex_research.utils.phase import read_checkpoint, replace_checkpoint, write_checkpoint

    trained, masters, optimizer = _trained_state()
    directory = tmp_path / "checkpoint"
    replace_checkpoint(directory, lambda path: write_checkpoint(path, 1, trained, masters, optimizer, {}))
    assert read_checkpoint(directory, tensors=False)["step"] == 1

    def interrupted(path):
        write_checkpoint(path, 2, trained, masters, optimizer, {})
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        replace_checkpoint(directory, interrupted)
    assert read_checkpoint(directory, tensors=False)["step"] == 1
    replace_checkpoint(directory, lambda path: write_checkpoint(path, 3, trained, masters, optimizer, {}))
    assert read_checkpoint(directory, tensors=False)["step"] == 3
    assert sorted(item.name for item in tmp_path.iterdir()) == ["checkpoint"]


def test_every_local_save_keeps_its_own_step_directory(tmp_path):
    from polyplex_research.utils.phase import read_checkpoint, save_local_checkpoint, write_checkpoint

    trained, masters, optimizer = _trained_state()
    for step in (250, 500):
        directory = save_local_checkpoint(
            tmp_path, step, lambda path: write_checkpoint(path, step, trained, masters, optimizer, {})
        )
        assert directory == tmp_path / f"checkpoint-{step}"
        assert read_checkpoint(directory, tensors=False)["step"] == step
    assert sorted(item.name for item in tmp_path.iterdir()) == ["checkpoint-250", "checkpoint-500"]


def test_a_resume_picks_the_highest_step_including_the_older_single_checkpoint(monkeypatch, tmp_path):
    import polyplex_research.utils.phase as phase

    monkeypatch.setattr(phase, "unstash_checkpoint", lambda run_name, directory: pytest.fail("the store is asked"))
    runs = tmp_path / "runs"
    for name, step in (("checkpoint", 1500), ("checkpoint-1000", 1000), ("checkpoint-750", 750)):
        (runs / "old" / name).mkdir(parents=True)
        (runs / "old" / name / "manifest.json").write_text(json.dumps({"step": step}))
    assert phase.resume_checkpoint("old", str(runs / "new")) == runs / "old" / "checkpoint"
    (runs / "old" / "checkpoint-2000.partial").mkdir()
    (runs / "old" / "checkpoint-2000.partial" / "manifest.json").write_text(json.dumps({"step": 2000}))
    assert phase.resume_checkpoint("old", str(runs / "new")) == runs / "old" / "checkpoint"
    (runs / "old" / "checkpoint-1750").mkdir()
    (runs / "old" / "checkpoint-1750" / "manifest.json").write_text(json.dumps({"step": 1750}))
    assert phase.resume_checkpoint("old", str(runs / "new")) == runs / "old" / "checkpoint-1750"


def test_flex_attention_keeps_packed_documents_apart_as_sdpa_does():
    from conftest import VOCAB_SIZE, build_polyplex

    from polyplex_research.utils.phase import use_flex_attention

    model = build_polyplex()
    torch.manual_seed(2)
    ids = torch.randint(0, VOCAB_SIZE, (1, 10))
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 4, 5]])
    with torch.no_grad():
        packed = model(input_ids=ids, position_ids=positions, use_cache=False).logits
        alone = model(input_ids=ids[:, 4:], position_ids=positions[:, 4:], use_cache=False).logits
        flex = use_flex_attention(model)(input_ids=ids, position_ids=positions, use_cache=False).logits
    assert model.config._attn_implementation == "flex_attention"
    torch.testing.assert_close(flex, packed, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(flex[:, 4:], alone, rtol=1e-5, atol=1e-5)


def test_cosine_restarts_reach_zero_at_every_save_and_restart_at_the_peak():
    """SPEC S13.4a: warmup as before, 0 at each multiple of `save_every`, 1 right after, resumable from the step."""
    import math

    from polyplex_research.utils.phase import cosine_restart_factor

    assert [cosine_restart_factor(step, 4, 10) for step in (1, 2, 4)] == [0.25, 0.5, 1.0]
    for save in (10, 20, 30, 250):
        assert cosine_restart_factor(save, 4, 10) == pytest.approx(0.0)
    assert cosine_restart_factor(11, 4, 10) == pytest.approx(0.5 * (1 + math.cos(math.pi / 10)))
    assert cosine_restart_factor(15, 4, 10) == pytest.approx(0.5)
    assert cosine_restart_factor(7, 4, 10) == pytest.approx(0.5)
    curve = [cosine_restart_factor(step, 4, 10) for step in range(1, 41)]
    resumed = [cosine_restart_factor(step, 4, 10) for step in range(17, 41)]
    assert resumed == curve[16:]


@pytest.mark.parametrize(
    "extra, expected",
    [
        (None, {}),
        ({"run_name": "dense", "wandb": "https://wandb.ai/team/polyplex/runs/abc123"}, {"id": "abc123", "resume": "allow"}),
        ({"run_name": "other", "wandb": "https://wandb.ai/team/polyplex/runs/abc123"}, {}),
        ({"run_name": "dense", "wandb": None}, {}),
    ],
)
def test_init_dense_wandb_run_continues_own_run(monkeypatch, extra, expected):
    import wandb

    from polyplex_research.utils import phase

    calls = []
    monkeypatch.setattr(wandb, "init", lambda **kwargs: calls.append(kwargs) or "run")
    assert phase.init_dense_wandb_run("dense", extra, config={"resume": "x"}) == "run"
    assert calls == [{"project": "polyplex", "name": "dense", "config": {"resume": "x"}, **expected}]


def test_dense_eval_main_maps_arguments_and_resolves_checkpoint(monkeypatch, tmp_path):
    from polyplex_research.utils import phase

    runs = tmp_path / "runs"
    for step in (500, 1000):
        (runs / "seven-languages" / f"checkpoint-{step}").mkdir(parents=True)
        (runs / "seven-languages" / f"checkpoint-{step}" / "manifest.json").write_text(json.dumps({"step": step}))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(phase, "evaluate_dense_checkpoint", lambda **kwargs: calls.append(kwargs) or {"step": 1000})
    result = phase.dense_eval_main(
        ["--checkpoint", "seven-languages", "--run-name", "seven-languages-eval", "--tokenizer", "tok",
         "--language-stream-from-embeddings", "--language-lora-rank", "8", "--samples", "8",
         "--language-model-samples", "--eval-rows", "32", "--training-tokens", "tokens.bin",
         "--full-precision-layers", "0,31", "--no-ablations", "--quantization", "none", "--release-dir", "out",
         "--sample-temperature", "0.8"]
    )
    assert result == {"step": 1000}
    kwargs = calls[0]
    assert kwargs["checkpoint"] == str(runs / "seven-languages" / "checkpoint-1000")
    assert kwargs["run_name"] == "seven-languages-eval"
    assert kwargs["tokenizer"] == "tok"
    assert kwargs["language_stream_from_text"] is False
    assert kwargs["language_lora_rank"] == 8
    assert kwargs["samples"] == 8
    assert kwargs["language_model_samples"] is True
    assert kwargs["eval_texts"] == 32
    assert kwargs["training_tokens"] == "tokens.bin"
    assert kwargs["full_precision_layers"] == (0, 31)
    assert kwargs["ablations"] == ()
    assert kwargs["quantization"] is None
    assert kwargs["language_ffn_layers"] is None
    assert kwargs["release_dir"] == "out"
    assert kwargs["sample_temperature"] == 0.8


def test_a_release_is_compared_with_the_model_it_was_saved_from_on_the_same_tokens(monkeypatch):
    import transformers

    from polyplex_research.utils import phase

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(character) % 7 for character in text]

    class Model(torch.nn.Module):
        def __init__(self, shift):
            super().__init__()
            self.config = type("Config", (), {"vocab_size": 7})()
            self.shift = shift
            self.seen = []

        def forward(self, input_ids, position_ids):
            self.seen.append((input_ids.tolist(), position_ids.tolist()))
            logits = torch.nn.functional.one_hot(input_ids % 7, 8).float()
            logits[..., 0] += self.shift
            return type("Output", (), {"logits": logits})()

    trained, release = Model(0.0), Model(0.25)
    loaded = []
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda path, **kwargs: loaded.append((path, kwargs)) or release
    )
    result = phase.compare_release(trained, "release", {"en": ["abcdef", "ghij"]}, Tokenizer(), "cpu", characters=3)
    assert loaded[0][0] == "release" and loaded[0][1]["trust_remote_code"] is True
    assert trained.seen == release.seen
    assert trained.seen[0] == ([[7, 6, 0, 1]], [[0, 1, 2, 3]])
    assert result == {"en": {"max_abs_logit_difference": 0.25, "argmax_agreement": 1.0}}


def test_dense_eval_main_scores_the_assembly_without_checkpoint(monkeypatch, tmp_path):
    from polyplex_research.utils import phase

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(phase, "evaluate_dense_checkpoint", lambda **kwargs: calls.append(kwargs) or {})
    phase.dense_eval_main(["--checkpoint", "none", "--run-name", "start"])
    assert calls[0]["checkpoint"] is None
    assert calls[0]["ablations"] == phase.DENSE_SCALE_ABLATIONS
    assert calls[0]["language_stream_from_text"] is True


def test_a_device_spec_can_take_several_cards(tmp_path):
    import letify

    from polyplex_research.utils.phase import resolve_device

    launcher = letify.Launcher(config=tmp_path, home=False)
    assert resolve_device(launcher, "local.CPU*2") == launcher.providers.local.CPU * 2


def _data_parallel_worker(rank, world, port, results):
    import os

    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.phase import (
        average_gradients,
        dense_trainable_parameters,
        train_dense,
    )

    from conftest import build_polyplex

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world)
    table = torch.nn.Embedding(6, 2, sparse=True)
    table(torch.tensor([rank, 3])).sum().backward()
    dense = torch.nn.Parameter(torch.zeros(3))
    dense.grad = torch.full((3,), float(rank + 1))
    average_gradients([table.weight, dense])
    torch.manual_seed(0)
    model = build_polyplex()
    names = dense_trainable_parameters(model)
    train_dense(
        model,
        dense_collate_fn(SilentProcessor(), model),
        train_dataset=_dense_rows() * 8,
        eval_sets={},
        reference={},
        steps=3,
        batch_size=4,
        micro_batch_size=1,
        learning_rate=1e-2,
        warmup_steps=1,
        eval_every=100,
    )
    parameters = dict(model.named_parameters())
    results[rank] = {
        "sparse": table.weight.grad.to_dense(),
        "dense": dense.grad.clone(),
        "trained": {name: parameters[name].detach().clone() for name in names},
    }
    torch.distributed.destroy_process_group()


def test_two_data_parallel_processes_train_as_one_process_does():
    import socket

    import torch.multiprocessing as mp

    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.phase import dense_trainable_parameters, train_dense

    from conftest import build_polyplex

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    results = mp.Manager().dict()
    mp.start_processes(_data_parallel_worker, args=(2, port, results), nprocs=2, start_method="spawn")
    expected_sparse = torch.zeros(6, 2)
    expected_sparse[0] += 0.5
    expected_sparse[1] += 0.5
    expected_sparse[3] += 1.0
    for rank in range(2):
        assert torch.allclose(results[rank]["sparse"], expected_sparse)
        assert torch.equal(results[rank]["dense"], torch.full((3,), 1.5))

    torch.manual_seed(0)
    model = build_polyplex()
    names = dense_trainable_parameters(model)
    train_dense(
        model,
        dense_collate_fn(SilentProcessor(), model),
        train_dataset=_dense_rows() * 8,
        eval_sets={},
        reference={},
        steps=3,
        batch_size=4,
        micro_batch_size=1,
        learning_rate=1e-2,
        warmup_steps=1,
        eval_every=100,
    )
    parameters = dict(model.named_parameters())
    for rank in range(2):
        for name in names:
            assert torch.allclose(results[rank]["trained"][name], parameters[name], atol=1e-5), name


def test_a_data_parallel_dense_run_fetches_its_input_files_before_starting_the_processes(monkeypatch):
    import polyplex_research.utils.phase as phase

    started = {}
    monkeypatch.setattr(phase, "fetch_input_file", lambda value, directory: f"/fetched/{value[len('store:'):]}")
    monkeypatch.setattr(phase, "dense_data_parallel", lambda args, gpus: started.update(args=args, gpus=gpus) or {})
    monkeypatch.delenv("RANK", raising=False)
    phase.dense_main(["--language-stream", "store:stream", "--gpus", "2", "--run-name", "x"])
    assert started == {"args": ["--language-stream", "/fetched/stream", "--run-name", "x"], "gpus": 2}
