from dataclasses import dataclass
from langchain_core.language_models.chat_models import BaseChatModel
from ..models import EnvironmentFeedback
from .environment import GroundedEnvironment

@dataclass
class ReflexionTrial:
    number: int
    attempt: str
    feedback: EnvironmentFeedback
    reflection: str | None = None

@dataclass
class ReflexionResult:
    success: bool
    output: str
    trials: list[ReflexionTrial]
    memory: list[str]

def reflexion(
    task: str,
    llm: BaseChatModel,
    environment: GroundedEnvironment,
    max_trials: int = 3,
    memory_size: int = 3,
) -> ReflexionResult:
    
    if max_trials < 1 or memory_size < 1:
        raise ValueError("max_trials and memory_size must be positive")
        
    memory: list[str] = []
    trials: list[ReflexionTrial] = []
    best_attempt = ""
    best_score = -1.0
    
    for number in range(1, max_trials + 1):
        recalled = "\n".join(f"- {item}" for item in memory[-memory_size:]) or "- No prior trials."
        
        response = llm.invoke([
            ("system", "You are an Iron Bridge Safety Approver in a Reflexion loop. Formulate a safe execution plan or approval decision."),
            ("human", f"""Task: {task}
Episodic memory from previous failed trials (Safety violations):
{recalled}

Produce the complete safe execution plan. Apply remembered lessons strictly without discussing them."""),
        ], temperature=0.2)
        
        attempt = response.content.strip()
        feedback = environment.evaluate(attempt)
        trial = ReflexionTrial(number=number, attempt=attempt, feedback=feedback)
        
        if feedback.score > best_score:
            best_attempt, best_score = attempt, feedback.score
            
        if feedback.success:
            trials.append(trial)
            return ReflexionResult(True, attempt, trials, memory[-memory_size:])
            
        response = llm.invoke([
            ("system", "Generate a concise first-person Reflexion memory, focusing purely on safety protocols and dependencies missed."),
            ("human", f"""Task: {task}
Failed plan:
{attempt}

Grounded Database/Safety feedback (score {feedback.score}):
{chr(10).join('- ' + item for item in feedback.details)}

State what safety check or dependency I missed and the specific strategy I should use next trial. Start with 'I'."""),
        ], temperature=0.2)
        
        reflection = response.content.strip()
        trial.reflection = reflection
        trials.append(trial)
        memory.append(reflection)
        
    return ReflexionResult(False, best_attempt, trials, memory[-memory_size:])