
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Iterator
from models.utils import add_speaker_to_lines

@dataclass
class Response:
    text: str
    selected_cs: list[str] = field(default_factory=list)
    original: str = ''
    input: str = ''

@dataclass
class Turn:
    uid: int
    sid: str
    utt: str
    cs: dict[str, str] = field(default_factory=dict)
    beam_cs: dict[str, list[str]] = field(default_factory=dict)
    response: dict[str, Response] = field(default_factory=dict)

@dataclass
class Dialogue:
    dialogue_id: str
    turns: list[Turn] = field(default_factory=list)
    on_terminal: bool = False

    def context(self, turn_id=None):
        if turn_id is None:
            turn_id = len(self.turns) - 1
        return add_speaker_to_lines([t.utt for t in self.turns[:turn_id + 1]])

    def turns_to_execute(self, speaker=None) -> Turn:
        if self.on_terminal:
            return [self.turns[-1]]
        else:
            return [t for t in self.turns if speaker is None or t.sid == speaker]

    def __len__(self):
        return len(self.turns)   

@dataclass
class Dialogues:
    collection: list[Dialogue] = field(default_factory=list)

    def __iter__(self) -> Iterator[Dialogue]:
        return iter(self.collection)
    
    def __getitem__(self, key):
        return self.collection[key]

    def __len__(self):
        return len(self.collection)
