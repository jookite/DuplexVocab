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
"""Reading the multi session conversation corpora."""

import json

from polyplex_research.datasets.multisession import iter_conversation_turns


def _write(directory, name, conversation, turns):
    record = {
        "multisessionInfo": {"multisessionID": conversation},
        "topicInfo": {"topicType": "일상"},
        "sessionInfo": [{"dialog": [{"speaker": speaker, "utterance": text} for speaker, text in turns]}],
    }
    (directory / name).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def test_conversation_turns_keep_both_speakers_and_drop_unreadable_ones(tmp_path):
    train = tmp_path / "Training"
    train.mkdir()
    _write(train, "a.json", "c1", [
        ("speaker1", "안녕하세요 저는 서울에 살아요."),
        ("speaker2", "반가워요  저도 서울이에요!"),
        ("speaker1", "ㅋㅋ 그렇군요"),
        ("speaker2", "네"),
    ])
    texts = [turn.text for turn in iter_conversation_turns(str(tmp_path), "train", min_words=2, max_words=25)]
    assert texts == ["안녕하세요 저는 서울에 살아요.", "반가워요 저도 서울이에요!"]


def _write_sessions(directory, name, conversation, sessions, wizard=None):
    record = {
        "multisessionInfo": {"multisessionID": conversation},
        "sessionInfo": [
            {"sessionID": f"{conversation}-{index}", "dialog": [{"speaker": s, "utterance": t} for s, t in turns]}
            for index, turns in enumerate(sessions)
        ],
    }
    if wizard is not None:
        record["personaInfo"] = {"wizardInfo": {"speakerType": wizard}, "apprenticeInfo": {"speakerType": "speaker1"}}
    (directory / name).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def test_a_dialogue_follows_the_wizard_and_keeps_the_other_side_as_turns_it_hears(tmp_path):
    from polyplex_research.datasets.multisession import iter_dialogues

    train = tmp_path / "Training"
    train.mkdir()
    _write_sessions(train, "a.json", "c1", [[
        ("speaker1", "여행 계획을 세우고 있어"),
        ("speaker2", "어디로 가실 예정인가요?"),
        ("speaker1", "제주도 ㅋㅋ"),
        ("speaker2", "제주도 좋지요. 언제 가세요?"),
    ]], wizard="speaker2")

    dialogues = list(iter_dialogues(str(tmp_path), "train"))
    assert [(turn.own, turn.text) for turn in dialogues[0]] == [
        (False, "여행 계획을 세우고 있어"),
        (True, "어디로 가실 예정인가요?"),
        (False, "제주도 ㅋㅋ"),
        (True, "제주도 좋지요. 언제 가세요?"),
    ]


def test_an_unusable_own_turn_ends_the_dialogue_there_and_starts_a_new_one(tmp_path):
    from polyplex_research.datasets.multisession import iter_dialogues

    train = tmp_path / "Training"
    train.mkdir()
    _write_sessions(train, "a.json", "c1", [[
        ("speaker2", "안녕하세요 반가워요"),
        ("speaker1", "네 안녕하세요"),
        ("speaker2", "ㅋㅋ 그렇군요"),
        ("speaker1", "오늘 뭐 하세요?"),
        ("speaker2", "책을 읽을 거예요."),
    ]], wizard="speaker2")

    dialogues = [[turn.text for turn in d] for d in iter_dialogues(str(tmp_path), "train")]
    assert dialogues == [["안녕하세요 반가워요", "네 안녕하세요"], ["오늘 뭐 하세요?", "책을 읽을 거예요."]]


def test_without_a_wizard_each_session_follows_one_speaker_chosen_by_its_id(tmp_path):
    from polyplex_research.datasets.multisession import iter_dialogues

    train = tmp_path / "Training"
    train.mkdir()
    turns = [("speaker1", "안녕하세요 저는 학생이에요"), ("speaker2", "반가워요 저는 회사원이에요")]
    _write_sessions(train, "a.json", "c1", [turns, turns, turns, turns])

    owners = []
    for dialogue in iter_dialogues(str(tmp_path), "train"):
        owners.append([turn.own for turn in dialogue])
        assert sum(turn.own for turn in dialogue) == 1
    assert len(owners) == 4 and {tuple(o) for o in owners} <= {(True, False), (False, True)}
