import time
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from planning.algorithms import dynamic_decomposition, tree_of_thoughts, reflexion
from planning.algorithms.environment import GroundedEnvironment

def run_evaluation():
    # Setup model
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in env.")
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0.1)
    env = GroundedEnvironment()
    
    # Fixed Test Suite
    test_cases = [
        {"id": "TC1", "query": "Worker 2 requested an excavator for a 6-foot trench in Type C soil.", "type": "dynamic_favored"},
        {"id": "TC2", "query": "Worker 1 wants to operate the mobile crane near power lines.", "type": "lookahead_favored"},
        {"id": "TC3", "query": "Approve scaffold request for Worker 2.", "type": "simple_deterministic"}
    ]
    
    print(f"{'Method':<20} | {'TC ID':<5} | {'Success':<8} | {'Latency (s)':<11} | {'Output Length (chars)'}")
    print("-" * 75)
    
    for case in test_cases:
        goal = case["query"]
        
        # 1. Evaluate Dynamic Decomposition
        t0 = time.perf_counter()
        try:
            dyn_hist = dynamic_decomposition(goal, llm, max_steps=4)
            dyn_success = "Yes" if dyn_hist else "No"
            dyn_chars = sum(len(res) for _, res in dyn_hist)
        except Exception as e:
            dyn_success = "Error"
            dyn_chars = 0
        dyn_lat = time.perf_counter() - t0
        print(f"{'Dynamic Decomp':<20} | {case['id']:<5} | {dyn_success:<8} | {dyn_lat:<11.2f} | {dyn_chars}")

        # 2. Evaluate Tree of Thoughts
        t0 = time.perf_counter()
        tot_result = tree_of_thoughts(goal, llm, depth=2, beam_width=2)
        tot_success = "Yes" if tot_result else "No"
        tot_chars = len(tot_result[0].state) if tot_result else 0
        tot_lat = time.perf_counter() - t0
        print(f"{'Tree of Thoughts':<20} | {case['id']:<5} | {tot_success:<8} | {tot_lat:<11.2f} | {tot_chars}")

        # 3. Evaluate Reflexion
        t0 = time.perf_counter()
        ref_result = reflexion(goal, llm, env, max_trials=3, memory_size=2)
        ref_success = "Yes" if ref_result.success else "No"
        ref_chars = len(ref_result.output)
        ref_lat = time.perf_counter() - t0
        print(f"{'Reflexion':<20} | {case['id']:<5} | {ref_success:<8} | {ref_lat:<11.2f} | {ref_chars}")
        
        print("-" * 75)

if __name__ == "__main__":
    run_evaluation()