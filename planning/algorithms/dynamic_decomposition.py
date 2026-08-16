from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

class DynamicDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    done: bool
    next_task: str

def dynamic_decomposition(goal: str, llm: BaseChatModel, max_steps: int = 6) -> list[tuple[str, str]]:
    history: list[tuple[str, str]] = []
    
    system_prompt = (
        "You are an adaptive safety planner for Iron Bridge Construction. "
        "You decompose heavy equipment requests into safety checks (e.g., certification, soil stability, power lines). "
        "Use prior observations to decide what comes next. "
        "Crucially: If a hazard is observed (e.g., 'Type C soil' or 'unstable'), your immediate next task MUST address it (e.g., 'Request shoring plan' or 'Deny request') before checking other normal prerequisites."
    )

    for step in range(max_steps):
        observation = "\n".join(f"{task}: {result}" for task, result in history) or "None"
        decision = llm.with_structured_output(
            DynamicDecision,
            method="json_schema",
        ).invoke([
            ("system", system_prompt),
            ("human", f"""Goal: {goal}
Completed work and observations:
{observation}

Decide the single best next task. Set done to true only when the goal is met or completely blocked (rejected).
When done is true, use an empty string for next_task."""),
        ], temperature=0.1)
        
        if decision.done:
            break
            
        task = decision.next_task.strip()
        if not task:
            raise ValueError(f"Dynamic planner omitted next_task at step {step + 1}")
            
        response = llm.invoke([
            ("system", "Execute the next adaptive sub-task using the observations provided. Act as an expert safety engineer."),
            ("human", f"Goal: {goal}\nNext task: {task}\nPrior observations:\n{observation}"),
        ], temperature=0.2)
        
        result = response.content
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")
        
        history.append((task, result.strip()))
        
    return history