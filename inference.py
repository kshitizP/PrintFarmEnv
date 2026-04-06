import os
import json
from openai import OpenAI
from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.models import FarmAction, FarmActionEnum

api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
api_key = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", ""))

client = OpenAI(
    base_url=api_base,
    api_key=api_key if api_key else "dummy_key"
)

def build_system_prompt() -> str:
    return """You are an automated Floor Manager for a 3D printing farm. 
Your goal is to manage incoming jobs, inventory, and hardware issues.
Respond strictly in JSON matching this schema:
{
  "action": "ASSIGN_JOB" | "SWAP_FILAMENT" | "CANCEL_JOB" | "WAIT",
  "printer_id": <int>,
  "job_id": "<string>",
  "material": "<string>"
}
If waiting, output {"action": "WAIT"}
"""

def extract_action(state_json: str) -> FarmAction:
    if not api_key:
        return FarmAction(action=FarmActionEnum.WAIT)
        
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": f"Current State: {state_json}"}
            ],
            response_format={ "type": "json_object" }
        )
        
        content = response.choices[0].message.content
        action_data = json.loads(content)
        return FarmAction(**action_data)
    except Exception as e:
        print(f"Error accessing LLM or parsing: {e}")
        return FarmAction(action=FarmActionEnum.WAIT)

def run_task(task_id: str, env: PrintFarmEnvironment) -> float:
    print(f"\n======================================")
    print(f"       STARTING {task_id.upper()}")
    print(f"======================================")
    observation = env.reset(task_id)
    done = False
    
    while not done:
        env.render_dashboard()
        action = extract_action(observation.model_dump_json())
        print(f"-> Agent Action: {action.action.value}")
        
        observation, reward, done, info = env.step_legacy(action)
        if info.get("error"):
            print(f"-> Action Error: {info['error']}")
            
        if done:
            print(f"\n{task_id.upper()} COMPLETED. Grader Reward: {reward}")
            env.render_dashboard()
            return reward
    return 0.0

if __name__ == "__main__":
    env = PrintFarmEnvironment()
    tasks = ["task_1", "task_2", "task_3"]
    total_reward = 0.0
    
    for t in tasks:
        total_reward += run_task(t, env)
        
    print(f"\nFinal Total Reward: {total_reward} / {len(tasks)}")
