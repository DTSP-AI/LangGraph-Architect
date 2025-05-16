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
from langchain.utilities import SerpAPIWrapper
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import LLMChain

from langgraph.graph import StateGraph, START, END

# ─── Memory Manager ─────────────────────────────────────────────────────────────
from memory_manager import (
    get_chat_history,
    clear_chat_history,         # re-introduced usage
    add_to_vector_memory,
    query_vector_memory
)

# ─── Logging & Env Setup ────────────────────────────────────────────────────────
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set")
if not SERPAPI_API_KEY:
    raise EnvironmentError("SERPAPI_API_KEY not set")

BASE_DIR = os.path.dirname(__file__)

# ─── Prompt & KB Loading ─────────────────────────────────────────────────────────
def load_prompt(path: str) -> Dict[str, str]:
    raw = json.load(open(path, encoding="utf-8"))
    system = "\n".join(raw["system"]) if isinstance(raw["system"], list) else raw["system"]
    user   = "\n".join(raw.get("user_template", [])) if isinstance(raw.get("user_template"), list) else raw.get("user_template", "")
    return {"system": system, "user": user}

supervisor_prompt = load_prompt(os.path.join(BASE_DIR, "prompts/supervisor.json"))
agent1_prompt     = load_prompt(os.path.join(BASE_DIR, "prompts/agent_1.json"))
agent2_prompt     = load_prompt(os.path.join(BASE_DIR, "prompts/agent_2.json"))

tools_kb = open(os.path.join(BASE_DIR, "knowledge_base/LGarchitect_tools_slim.txt"), encoding="utf-8").read()
multi_kb = open(os.path.join(BASE_DIR, "knowledge_base/LGarchitect_multi_agent_slim.txt"), encoding="utf-8").read()
core_kb  = open(os.path.join(BASE_DIR, "knowledge_base/LGarchitect_LangGraph_core_slim.txt"), encoding="utf-8").read()

# ─── Data Models ────────────────────────────────────────────────────────────────
class ClientIntake(BaseModel):
    ClientProfile:   dict
    SalesOps:        dict
    Marketing:       dict
    Retention:       dict
    AIReadiness:     dict
    TechStack:       dict
    GoalsTimeline:   dict
    HAF:             dict
    CII:             dict
    ReferenceDocs:   str

class IntakeSummary(BaseModel):
    ClientProfile:   dict
    Good:            List[str]
    Bad:             List[str]
    Ugly:            List[str]
    SolutionSummary: str
    WorkflowOutline: List[str]
    HAF:             dict
    CII:             dict

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
    error:         Dict[str, Any]

# ─── LLM & Tools Initialization ────────────────────────────────────────────────
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, openai_api_key=OPENAI_API_KEY)
search_tool = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)

def strip_fences(text: str) -> str:
    text = re.sub(r"^```[\w]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```$", "", text, flags=re.MULTILINE)
    return text.strip()

def robust_invoke(messages: List[Any]) -> Any:
    for attempt in range(3):
        try:
            return llm.invoke(messages)
        except Exception as e:
            logger.warning(f"LLM invoke failed (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
    logger.error("LLM invoke permanently failed after 3 attempts", exc_info=True)
    raise

# ─── Node 1: Supervisor Agent ───────────────────────────────────────────────────
def supervisor_node(state: GraphState) -> dict:
    try:
        intake_model = state["intake"]
        session_id   = intake_model.ClientProfile.get("name", "default")

        # **New**: if the client JSON asks to reset, wipe history
        if intake_model.ClientProfile.get("reset_history", False):
            clear_chat_history(session_id)

        history = get_chat_history(session_id)
        msgs: List[Any] = [ SystemMessage(content=supervisor_prompt["system"]) ]
        msgs.extend(history.messages)

        raw_json = json.dumps(intake_model.model_dump(), indent=2)
        msgs.append(HumanMessage(content=supervisor_prompt["user"].replace("{RAW_INTAKE_JSON}", raw_json)))
        for doc in query_vector_memory(raw_json, k=3):
            msgs.append(SystemMessage(content=f"MemoryContext: {doc.page_content}"))

        resp = robust_invoke(msgs)
        content = strip_fences(resp.content)

        history.add_user_message(raw_json)
        history.add_ai_message(content)
        add_to_vector_memory(content, metadata={"session_id": session_id})

        parsed   = json.loads(content)
        validated = parsed.get("validated_intake", parsed)
        intake_model = ClientIntake(**validated)
        return {"intake": intake_model}

    except Exception as e:
        logger.error(f"[SUPERVISOR] {e}", exc_info=True)
        return {"error": {"node": "supervisor", "message": str(e)}}

# ─── Node 2: Web Search Agent ────────────────────────────────────────────────────
def websearch_node(state: GraphState) -> dict:
    try:
        industry = state["intake"].ClientProfile.get("industry", "")
        summary  = search_tool.run(f"Top AI automation best practices in {industry}")
        return {"websummary": summary}
    except Exception as e:
        logger.error(f"[WEBSEARCH] {e}", exc_info=True)
        return {"error": {"node": "websearch", "message": str(e)}}

# ─── Node 3: Solution Summarizer Agent ───────────────────────────────────────────
def summarizer_node(state: GraphState) -> dict:
    try:
        raw_json   = json.dumps(state["intake"].model_dump(), indent=2)
        system_ctx = f"{agent1_prompt['system']}\n\nWebSummary:\n{state.get('websummary','')}\n\n{multi_kb}"
        user_ctx   = agent1_prompt["user"].replace("{RAW_INTAKE_JSON}", raw_json)

        resp    = robust_invoke([
            SystemMessage(content=system_ctx),
            HumanMessage(content=user_ctx)
        ])
        content = strip_fences(resp.content)
        summary = IntakeSummary.model_validate_json(json.dumps(json.loads(content)))
        return {"summary": summary}
    except Exception as e:
        logger.error(f"[SUMMARIZER] {e}", exc_info=True)
        return {"error": {"node": "summarizer", "message": str(e)}}

# ─── Node 4: Report Generation Agent ─────────────────────────────────────────────
def report_node(state: GraphState) -> dict:
    try:
        summary_json = json.dumps(state["summary"].model_dump(), indent=2)
        system_ctx   = f"{agent2_prompt['system']}\n\n{core_kb}\n{tools_kb}"
        user_ctx     = agent2_prompt["user"].replace("{SUMMARY_JSON}", summary_json)

        resp    = robust_invoke([
            SystemMessage(content=system_ctx),
            HumanMessage(content=user_ctx)
        ])
        content = strip_fences(resp.content)
        data    = json.loads(content)
        return {
            "client_report": ClientFacingReport(report_markdown=data["client_report"]),
            "dev_report":    DevFacingReport(blueprint_graph=data["developer_report"])
        }
    except Exception as e:
        logger.error(f"[REPORT] {e}", exc_info=True)
        return {"error": {"node": "report", "message": str(e)}}

# ─── Build LangGraph Pipeline ────────────────────────────────────────────────────
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

# ─── Public API ─────────────────────────────────────────────────────────────────
def run_pipeline(raw_intake: dict) -> Any:
    intake_model = ClientIntake(**raw_intake)
    result       = graph.invoke({"intake": intake_model})
    return {
        "client_report": result.get("client_report", {}).report_markdown,
        "dev_report":    result.get("dev_report", {}).blueprint_graph,
        "error":         result.get("error", None)
    }

# ─── Supervisor Chain for UI Q&A ─────────────────────────────────────────────────
def create_supervisor_chain() -> LLMChain:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(supervisor_prompt["system"]),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template(supervisor_prompt["user"]),
    ])
    return LLMChain(llm=llm, prompt=prompt)

supervisor_chain = create_supervisor_chain()

# ─── CLI Test Hook ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample  = json.load(open(os.path.join(BASE_DIR, "sample_intake.json"), encoding="utf-8"))
    reports = run_pipeline(sample)
    print("\n==== CLIENT REPORT ====\n", reports.get("client_report"))
    print("==== DEV REPORT ====\n",     reports.get("dev_report"))
