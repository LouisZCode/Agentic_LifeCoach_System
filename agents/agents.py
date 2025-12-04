"""
Here you will find all the agents form create_agent, already with their prompts loaded
To edit prompts, check the prompts.yaml document
"""

from functions import read_client_folder
from langchain.agents import create_agent
import yaml
from dotenv import load_dotenv
import os

load_dotenv()

def load_prompts():
    with open("agents/prompts.yaml", "r", encoding="UTF-8") as f:
        prompts = yaml.safe_load(f)
    return prompts

prompts = load_prompts()
first_draft_agent_prompt = prompts["class0_prompt"]


first_draft_agent = create_agent(
    system_prompt=first_draft_agent_prompt,
    model="google_genai:gemini-2.5-flash",
    tools=[read_client_folder]
)
