"""
State graph engine with:
  - durable checkpoint after every node
  - HITL pause (expected human decision)
  - failure tickets (unplanned errors) with resume from checkpoint
  - crash recovery: load latest checkpoint and continue

Graphs are cyclic-capable (can loop back); distinct from planning DAGs.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from platform_db import store


class HITLPause(Exception):
    """Raised when a node requires human approval. Engine opens HITL task and stops."""

    def __init__(self, title: str, reason: str, payload: dict):
        self.title = title
        self.reason = reason
        self.payload = payload
        super().__init__(reason)


class NodeFailure(Exception):
    """Unplanned failure inside a node — becomes a failure ticket."""

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(message)


@dataclass
class GraphState:
    """Mutable run state persisted at every checkpoint."""

    run_id: str
    graph_name: str
    worker_id: Optional[str]
    current_node: str
    step_index: int = 0
    data: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    hitl_decision: Optional[str] = None
    hitl_note: Optional[str] = None
    status: str = "running"

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "graph_name": self.graph_name,
            "worker_id": self.worker_id,
            "current_node": self.current_node,
            "step_index": self.step_index,
            "data": self.data,
            "history": self.history,
            "hitl_decision": self.hitl_decision,
            "hitl_note": self.hitl_note,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GraphState":
        return cls(
            run_id=d["run_id"],
            graph_name=d["graph_name"],
            worker_id=d.get("worker_id"),
            current_node=d["current_node"],
            step_index=d.get("step_index", 0),
            data=d.get("data") or {},
            history=d.get("history") or [],
            hitl_decision=d.get("hitl_decision"),
            hitl_note=d.get("hitl_note"),
            status=d.get("status", "running"),
        )


NodeFn = Callable[[GraphState], str]  # returns next node id or "END"


@dataclass
class StateGraph:
    name: str
    start: str
    nodes: dict[str, NodeFn]
    # optional: conditions that force HITL before node runs
    hitl_guards: dict[str, Callable[[GraphState], Optional[tuple[str, str, dict]]]] = field(
        default_factory=dict
    )

    def run(
        self,
        input_data: dict,
        worker_id: Optional[str] = None,
        max_steps: int = 40,
    ) -> GraphState:
        run_id = store.create_run(self.name, worker_id, input_data)
        state = GraphState(
            run_id=run_id,
            graph_name=self.name,
            worker_id=worker_id,
            current_node=self.start,
            data=dict(input_data),
        )
        return self._execute(state, max_steps=max_steps)

    def resume(self, run_id: str, max_steps: int = 40) -> GraphState:
        """Resume from latest checkpoint (after HITL resolve or ticket resolve)."""
        cp = store.latest_checkpoint(run_id)
        if not cp:
            raise ValueError(f"No checkpoint for run {run_id}")
        state = GraphState.from_dict(cp["state"])
        # If paused on HITL, advance past the node that paused once decision is set
        if state.status == "paused_hitl" and state.hitl_decision:
            state.status = "running"
            store.update_run_status(run_id, "running")
        elif state.status == "failed":
            # ticket resolved — retry current node
            state.status = "running"
            store.update_run_status(run_id, "running")
        return self._execute(state, max_steps=max_steps)

    def _execute(self, state: GraphState, max_steps: int) -> GraphState:
        steps = 0
        while state.current_node != "END" and steps < max_steps:
            steps += 1
            node_id = state.current_node
            if node_id not in self.nodes:
                self._fail(state, "unknown_node", f"Node {node_id} not defined")
                return state

            # HITL guard before node
            guard = self.hitl_guards.get(node_id)
            if guard and not state.hitl_decision:
                need = guard(state)
                if need:
                    title, reason, payload = need
                    self._checkpoint(state)
                    task_id = store.open_hitl(
                        state.run_id, node_id, title, reason, payload
                    )
                    state.status = "paused_hitl"
                    state.data["pending_hitl_task"] = task_id
                    state.history.append({"node": node_id, "event": "hitl_pause", "task_id": task_id})
                    self._checkpoint(state)
                    return state

            try:
                next_node = self.nodes[node_id](state)
                state.history.append(
                    {
                        "node": node_id,
                        "event": "ok",
                        "next": next_node,
                        "hitl": state.hitl_decision,
                    }
                )
                # clear one-shot HITL decision after consumption
                if state.hitl_decision:
                    state.data["last_hitl_decision"] = state.hitl_decision
                    state.data["last_hitl_note"] = state.hitl_note
                    state.hitl_decision = None
                    state.hitl_note = None
                state.current_node = next_node
                state.step_index += 1
                self._checkpoint(state)
            except HITLPause as e:
                self._checkpoint(state)
                task_id = store.open_hitl(
                    state.run_id, node_id, e.title, e.reason, e.payload
                )
                state.status = "paused_hitl"
                state.data["pending_hitl_task"] = task_id
                state.history.append({"node": node_id, "event": "hitl_pause", "task_id": task_id})
                self._checkpoint(state)
                return state
            except NodeFailure as e:
                self._fail(state, e.error_type, e.message)
                return state
            except Exception as e:
                self._fail(state, "exception", f"{e}\n{traceback.format_exc()}")
                return state

        if state.current_node == "END":
            state.status = "completed"
            store.update_run_status(state.run_id, "completed", state.data)
            self._checkpoint(state)
        return state

    def _checkpoint(self, state: GraphState) -> str:
        return store.save_checkpoint(
            state.run_id, state.current_node, state.step_index, state.to_dict()
        )

    def _fail(self, state: GraphState, error_type: str, message: str) -> None:
        cid = self._checkpoint(state)
        tid = store.open_ticket(
            state.run_id, state.current_node, error_type, message, cid
        )
        state.status = "failed"
        state.data["failure_ticket"] = tid
        state.history.append(
            {"node": state.current_node, "event": "failure", "ticket_id": tid}
        )
        self._checkpoint(state)


def apply_hitl_to_state(run_id: str, decision: str, note: str = "") -> GraphState:
    """After admin resolves HITL in platform, inject decision into latest checkpoint state."""
    cp = store.latest_checkpoint(run_id)
    if not cp:
        raise ValueError("no checkpoint")
    state = GraphState.from_dict(cp["state"])
    state.hitl_decision = decision
    state.hitl_note = note
    state.status = "running"
    store.save_checkpoint(run_id, state.current_node, state.step_index, state.to_dict())
    store.update_run_status(run_id, "running")
    return state
