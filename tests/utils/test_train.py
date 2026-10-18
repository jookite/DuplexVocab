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
"""What the base trainer guarantees when a run is asked to stop."""

import os
import signal

import pytest
from transformers import Trainer as HuggingFaceTrainer
from transformers import TrainerControl

from polyplex_research.utils.train import GracefulShutdownCallback, Trainer


@pytest.fixture
def trainer(monkeypatch):
    r"""
    A trainer whose loop is a stub, so a test drives the signal path without a model or accelerate.
    """
    instance = object.__new__(Trainer)
    instance.callbacks = []
    monkeypatch.setattr(Trainer, "add_callback", lambda self, c: self.callbacks.append(c))
    monkeypatch.setattr(Trainer, "remove_callback", lambda self, c: self.callbacks.remove(c))
    return instance


def test_callback_raises_the_flags_only_after_a_signal():
    r"""
    The flags are what the loop reads, and raising them unasked would stop a healthy run.
    """
    callback = GracefulShutdownCallback()
    control = TrainerControl()

    callback.on_step_end(None, None, control)
    assert not control.should_training_stop and not control.should_save

    callback.requested = True
    callback.on_step_end(None, None, control)
    assert control.should_training_stop and control.should_save


def test_signal_during_training_reaches_the_callback(trainer, monkeypatch):
    r"""
    The handler only records, so what proves it works is the callback the loop asks at its next step.
    """
    seen = {}

    def loop(self, *args, **kwargs):
        os.kill(os.getpid(), signal.SIGINT)
        seen["requested"] = self.callbacks[-1].requested
        return "finished"

    monkeypatch.setattr(HuggingFaceTrainer, "train", loop)
    assert trainer.train() == "finished"
    assert seen["requested"] is True


def test_handlers_are_restored_and_the_callback_removed(trainer, monkeypatch):
    r"""
    The handlers outlive the loop only if `train` leaks them, and a leaked one would swallow the
    interrupt of whatever runs after the phase.
    """
    before = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}

    def loop(self, *args, **kwargs):
        assert signal.getsignal(signal.SIGINT) is not before[signal.SIGINT]
        raise RuntimeError("the loop failed")

    monkeypatch.setattr(HuggingFaceTrainer, "train", loop)
    with pytest.raises(RuntimeError):
        trainer.train()

    assert {number: signal.getsignal(number) for number in before} == before
    assert trainer.callbacks == []


def test_a_second_signal_falls_through_to_the_previous_handler(trainer, monkeypatch):
    r"""
    A run that stops responding has to stay killable, so escalation is handed back to whoever owned
    the signal before the loop took it.
    """
    seen = {}
    previous = signal.signal(signal.SIGTERM, lambda number, frame: seen.setdefault("number", number))

    def loop(self, *args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGTERM)
        return "finished"

    monkeypatch.setattr(HuggingFaceTrainer, "train", loop)
    try:
        trainer.train()
    finally:
        signal.signal(signal.SIGTERM, previous)

    assert seen["number"] == signal.SIGTERM


def test_the_research_tree_callback_reports_the_last_evaluation_and_the_wandb_link(monkeypatch):
    r"""
    The experiment pull request is the lab note, so a finished run writes its final numbers and the
    link to its curves there, and nothing else.
    """
    from transformers import TrainerControl, TrainerState

    from polyplex_research.utils.train import ResearchTreeCallback

    logged, fields = {}, {}
    callback = ResearchTreeCallback(log=logged.update, set_fields=fields.update, wandb_url=lambda: "https://wandb.ai/x/y/runs/z")
    state, control = TrainerState(), TrainerControl()
    callback.on_evaluate(None, state, control, metrics={"eval_all_loss": 5.5, "epoch": 0.5})
    callback.on_evaluate(None, state, control, metrics={"eval_all_loss": 5.2, "eval_new_loss": 5.8, "epoch": 1.0})
    callback.on_train_end(None, state, control)

    assert logged == {"eval_all_loss": 5.2, "eval_new_loss": 5.8}
    assert fields == {"wandb": "https://wandb.ai/x/y/runs/z"}


def test_the_research_tree_callback_stays_silent_on_other_ranks():
    from transformers import TrainerControl, TrainerState

    from polyplex_research.utils.train import ResearchTreeCallback

    logged = {}
    callback = ResearchTreeCallback(log=logged.update, set_fields=logged.update, wandb_url=lambda: "url")
    state = TrainerState(is_world_process_zero=False)
    callback.on_evaluate(None, state, TrainerControl(), metrics={"eval_all_loss": 5.2})
    callback.on_train_end(None, state, TrainerControl())

    assert logged == {}


def test_the_run_record_reaches_the_wandb_config_once_training_begins():
    from types import SimpleNamespace

    from transformers import TrainerControl, TrainerState

    from polyplex_research.utils.train import RunRecordCallback

    updates = []
    run = SimpleNamespace(config=SimpleNamespace(update=lambda values, allow_val_change: updates.append(values)))
    callback = RunRecordCallback({"record/git_commit": "abc1234", "phase/train_rows": False}, active_run=lambda: run)
    callback.on_train_begin(None, TrainerState(), TrainerControl())
    assert updates == [{"record/git_commit": "abc1234", "phase/train_rows": False}]

    silent = RunRecordCallback({"a": 1}, active_run=lambda: None)
    silent.on_train_begin(None, TrainerState(), TrainerControl())


def test_the_git_record_names_the_commit_and_branch_of_the_checkout():
    import subprocess

    from polyplex_research.utils.train import git_record

    record = git_record()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert record["record/git_commit"] == head
    assert record["record/git_branch"]
    assert isinstance(record["record/git_dirty"], bool)
