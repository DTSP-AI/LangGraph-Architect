# graph.py

import os
import json
import logging
import re
import time
from typing import TypedDict, Any, List, Dict

from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import SerpAPIWrapper
from langgraph.graph import StateGraph, START, END

# ─── Memory Manager ─────────────────────────────────────────────────────────────
from memory_manager import (
    get_chat_history,
    clear_chat_history,
    add_to_vector_memory,
    query_vector_memory
)

##############################################
# Environment Setup
##############################################
load_dotenv()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY  = os.getenv("SERPAPI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set")
if not SERPAPI_API_KEY:
    raise EnvironmentError("SERPAPI_API_KEY not set")

BASE_DIR = os.path.dirname(__file__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##############################################
# Load Prompt Content & Knowledge Bases
##############################################
with open(os.path.join(BASE_DIR, "prompts/supervisor.json"), encoding="utf-8") as f:
    supervisor_prompt = json.load(f)
with open(os.path.join(BASE_DIR, "prompts/agent_1.json"), encoding="utf-8") as f:
    agent1_prompt = json.load(f)
with open(os.path.join(BASE_DIR, "prompts/agent_2.json"), encoding="utf-8") as f:
    agent2_prompt = json.load(f)

with open(os.path.join(BASE_DIR, "knowledge_base/LGarchitect_tools_slim.txt"), encoding="utf-8") as f:
    tools_kb = f.read()
with open(os.path.join(BASE_DIR, "knowledge_base/LGarchitect_multi_agent_slim.txt"), encoding="utf-8") as f:
    multi_kb = f.read()
with open(os.path.join(BASE_DIR, "knowledge_base/LGarchitect_LangGraph_core_slim.txt"), encoding="utf-8") as f:
    core_kb = f.read()

##############################################
# Data Models
##############################################
class ClientIntake(BaseModel):
    ClientProfile: Dict[str, Any]
    SalesOps:      Dict[str, Any]
    Marketing:     Dict[str, Any]
    Retention:     Dict[str, Any]
    AIReadiness:   Dict[str, Any]
    TechStack:     Dict[str, Any]
    GoalsTimeline: Dict[str, Any]
    HAF:           Dict[str, Any]
    CII:           Dict[str, Any]
    ReferenceDocs: str

class IntakeSummary(BaseModel):
    ClientProfile:   Dict[str, Any]
    Good:            List[str]
    Bad:             List[str]
    Ugly:            List[str]
    SolutionSummary: str
    WorkflowOutline: List[str]
    HAF:             Dict[str, Any]
    CII:             Dict[str, Any]

class ClientFacingReport(BaseModel):
    report_markdown: str

class DevFacingReport(BaseModel):
    blueprint_graph: str

class GraphState(TypedDict):
    intake:        ClientIntake
    websummary:    str
    summary:       IntakeSummary
    client_report: ClientFacingReport
    dev_report:    DevFacingReport

##############################################
# LLM & Tool Initialization
##############################################
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, openai_api_key=OPENAI_API_KEY)
search_tool = SerpAPIWrapper(api_key=SERPAPI_API_KEY)

##############################################
# Helpers
##############################################
def strip_fences(text: str) -> str:
    """Remove markdown fences."""
    text = re.sub(r"^```[\w]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```$", "", text, flags=re.MULTILINE)
    return text.strip()

def robust_invoke(messages: List[Any]) -> Any:
    """Invoke LLM with retries."""
    for attempt in range(3):
        try:
            return llm.invoke(messages)
        except Exception as e:
            logger.warning(f"LLM invoke failed (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
    logger.error("LLM invoke permanently failed after 3 attempts", exc_info=True)
    raise

##############################################
# Node 1: Supervisor Agent
##############################################
def supervisor_node(state: GraphState) -> dict:
    """
    Supervisor Agent:
      - Loads per-client chat history
      - Injects static prompt + intake JSON
      - Retrieves relevant docs from vector memory
      - Calls LLM, updates history & vector memory
      - Validates/enriches intake for next node
    """
    # derive session_id from client name
    session_id = state["intake"].ClientProfile.get("name", "default")
    history = get_chat_history(session_id)

    # build messages: system -> history -> new user input
    messages: List[Any] = [
        SystemMessage(content=supervisor_prompt["system"])
    ]
    # replay chat history
    for msg in history.messages:
        messages.append(msg)
    # include static intake JSON
    raw_intake = json.dumps(state["intake"].model_dump(), indent=2)
    user_msg = SystemMessage(content=supervisor_prompt["user_template"].replace("{RAW_INTAKE_JSON}", raw_intake))
    messages.append(user_msg)

    # optionally retrieve vector memory for context
    docs = query_vector_memory(raw_intake, k=3)
    for doc in docs:
        messages.append(SystemMessage(content=f"MemoryContext: {doc.page_content}"))

    # invoke LLM
    resp = robust_invoke(messages)
    content = strip_fences(resp.content)

    # record assistant reply to history & vector memory
    history.add_user_message(raw_intake)
    history.add_ai_message(content)
    add_to_vector_memory(content, metadata={"session_id": session_id})

    # parse validated intake JSON
    parsed = json.loads(content)
    validated = parsed.get("validated_intake", parsed)
    intake_model = ClientIntake(**validated)
    logger.info("[SUPERVISOR] Intake validated/enriched")
    return {"intake": intake_model}

##############################################
# Node 2: Web Search Agent
##############################################
def websearch_node(state: GraphState) -> dict:
    """
    Web Search Agent:
      - Queries best practices for client's industry
    """
    industry = state["intake"].ClientProfile.get("industry", "")
    query = f"Top AI automation best practices in {industry}"
    logger.info(f"[WEBSEARCH] Querying web: {query}")
    results = search_tool.run(query)
    return {"websummary": results}

##############################################
# Node 3: Solution Summarizer Agent
##############################################
def summarizer_node(state: GraphState) -> dict:
    """
    Solution Summarizer:
      - Merges intake + web summary
      - Produces Good/Bad/Ugly, tech stack, workflows
    """
    raw_json = json.dumps(state["intake"].model_dump(), indent=2)
    web_summary = state.get("websummary", "")
    system_ctx = agent1_prompt["system"] + "\n\nWebSummary:\n" + web_summary + "\n\n" + multi_kb
    user_ctx = agent1_prompt["user_template"].replace("{RAW_INTAKE_JSON}", raw_json)

    resp = robust_invoke([
        SystemMessage(content=system_ctx),
        HumanMessage(content=user_ctx)
    ])
    content = strip_fences(resp.content)
    summary = IntakeSummary.parse_raw(content)
    logger.info("[SUMMARIZER] Parsed IntakeSummary")
    return {"summary": summary}

##############################################
# Node 4: Report Generation Agent
##############################################
def report_node(state: GraphState) -> dict:
    """
    Report Generator:
      - Builds client-facing markdown & dev blueprint
    """
    summary_json = json.dumps(state["summary"].model_dump(), indent=2)
    system_ctx = agent2_prompt["system"] + "\n\n" + core_kb + "\n" + tools_kb
    user_ctx = agent2_prompt["user_template"].replace("{SUMMARY_JSON}", summary_json)

    resp = robust_invoke([
        SystemMessage(content=system_ctx),
        HumanMessage(content=user_ctx)
    ])
    content = strip_fences(resp.content)
    data = json.loads(content)

    logger.info("[REPORT] Parsed reports JSON")
    return {
        "client_report": ClientFacingReport(report_markdown=data["client_report"]),
        "dev_report":    DevFacingReport(blueprint_graph=data["developer_report"])
    }

##############################################
# Build LangGraph Pipeline
##############################################
builder = StateGraph(GraphState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("websearch",  websearch_node)
builder.add_node("summarize",  summarizer_node)
builder.add_node("report",     report_node)

builder.add_edge(START,        "supervisor")
builder.add_edge("supervisor", "websearch")
builder.add_edge("websearch",  "summarize")
builder.add_edge("summarize",  "report")
builder.add_edge("report",     END)

graph = builder.compile()

##############################################
# Public API
##############################################
def run_pipeline(raw_intake: dict) -> Dict[str, str]:
    intake_model = ClientIntake(**raw_intake)
    result = graph.invoke({"intake": intake_model})
    return {
        "client_report": result["client_report"].report_markdown,
        "dev_report":    result["dev_report"].blueprint_graph
    }

##############################################
# CLI Test Hook
##############################################
if __name__ == "__main__":
    sample_file = os.path.join(BASE_DIR, "sample_intake.json")
    with open(sample_file, encoding="utf-8") as f:
        sample = json.load(f)
    reports = run_pipeline(sample)
    print("\n==== CLIENT REPORT ====\n", reports["client_report"])
    print("\n==== DEV REPORT ====\n",    reports["dev_report"])
