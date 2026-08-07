"""
Short-Term (Working) Memory for the Iron Bridge agent.

Two separate concerns live here on purpose:
  - `messages`: the rolling transcript (user/assistant/tool turns).
    This is what context_eval/'s four pruning strategies operate on.
  - `scratchpad`: the agent's *current working state* (plan, sub-goal,
    temp variables). It is a dict, not a message, so no pruning strategy
    in context_eval/ can accidentally delete it while trimming the
    transcript — that's the whole reason it's a separate object instead
    of "just another message in the list".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    worker_id: Optional[int] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "worker_id": self.worker_id,
            "created_at": self.created_at,
        }


class ShortTermMemory:
    """Rolling message buffer + scratchpad for one active session."""

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self.messages: list[Message] = []
        self.scratchpad: dict[str, Any] = {
            "plan": None,
            "current_subgoal": None,
            "working_vars": {},
        }

    # ---------------- transcript ----------------
    def add(
        self, role: Role, content: str, worker_id: Optional[int] = None
    ) -> Optional[Message]:
        """Append a turn. Returns the evicted (oldest) message if the
        buffer was over max_turns, so the caller (memory/router.py) can
        decide its fate — this method never discards silently."""
        self.messages.append(Message(role=role, content=content, worker_id=worker_id))
        evicted: Optional[Message] = None
        if len(self.messages) > self.max_turns:
            evicted = self.messages.pop(0)
        return evicted

    def get_context(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self.messages]

    def is_over_budget(self) -> bool:
        return len(self.messages) > self.max_turns

    # ---------------- scratchpad ----------------
    def update_plan(
        self,
        plan: Optional[str] = None,
        subgoal: Optional[str] = None,
        **working_vars: Any,
    ) -> None:
        if plan is not None:
            self.scratchpad["plan"] = plan
        if subgoal is not None:
            self.scratchpad["current_subgoal"] = subgoal
        if working_vars:
            self.scratchpad["working_vars"].update(working_vars)

    def get_scratchpad(self) -> dict[str, Any]:
        return self.scratchpad

    def clear_transcript_keep_scratchpad(self) -> None:
        """Used by context_eval/ strategies that wipe the transcript
        (e.g. a hard sliding-window reset) — the scratchpad must survive
        this call untouched, which is the property every strategy in
        context_eval/ is tested against."""
        self.messages = []