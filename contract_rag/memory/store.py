from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Tuple

Role = Literal["user", "assistant", "system"]
Message = Tuple[Role, str]


@dataclass
class MemoryStore:
    messages: List[Message] = field(default_factory=list)

    def add(self, role: Role, content: str) -> None:
        self.messages.append((role, content))

    def add_user(self, content: str) -> None:
        self.add("user", content)

    def add_assistant(self, content: str) -> None:
        self.add("assistant", content)

    def last_k(self, k: int = 6) -> List[Message]:
        return self.messages[-k:]
