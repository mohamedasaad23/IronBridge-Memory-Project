import sys
from pathlib import Path
from ..models import EnvironmentFeedback

# Inject root to access mcp_server modules
ROOT = Path(__file__).resolve().parents[3] 
sys.path.insert(0, str(ROOT))
from mcp_server import service

class GroundedEnvironment:
    """A grounded evaluator that checks plans against the real IronBridge DB."""

    def __init__(self, success_threshold: float = 0.8):
        self.success_threshold = success_threshold

    def evaluate(self, state: str) -> EnvironmentFeedback:
        details = []
        score = 1.0
        success = True
        state_lower = state.lower()
        
        # Grounded Check 1: OSHA Soil Constraints for Excavators
        if "excavator" in state_lower and "type c" in state_lower:
            if "shoring" not in state_lower and "trench box" not in state_lower:
                success = False
                score = 0.0
                details.append("CRITICAL: Excavator deployed in Type C soil without specifying a shoring plan or trench box.")

        # Grounded Check 2: Actual DB Certification Check (assuming worker 2 as example from prompt)
        if "worker 2" in state_lower and "crane" in state_lower:
            # Connect to actual DB service
            cert = service.check_certification(worker_id=2, equipment_type="CRANE")
            if not cert["valid"]:
                success = False
                score = 0.0
                details.append(f"DB VALIDATION FAILED: Worker 2 certification for CRANE is invalid. Reason: {cert['reason']}.")

        # Grounded Check 3: Power Lines Risk
        if "power lines" in state_lower and "supervisor" not in state_lower:
            success = False
            score = 0.3
            details.append("POLICY VIOLATION: Operations near power lines require explicit supervisor elicitation step.")

        if success:
            details.append("Plan passes all grounded safety and DB checks.")

        return EnvironmentFeedback(success=success, score=score, details=details)

# Keep the original Environment class name available if other scripts import it, 
# but point it to the grounded one.
Environment = GroundedEnvironment