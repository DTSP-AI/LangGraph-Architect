import os
import json
import logging
import re
import time
from typing import TypedDict, Any, List, Dict, Optional # Added Optional

from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
# Use BaseMessage for type hinting, SystemMessage, HumanMessage for constructing
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_community.utilities import SerpAPIWrapper
from langgraph.graph import StateGraph, START, END

# Ensure memory_manager functions are correctly imported
from memory_manager import (
    get_chat_history,
    clear_chat_history, # Will reconsider its use in supervisor_node
    add_to_vector_memory,
    query_vector_memory,
    add_lc_message_to_history # If supervisor_node needs to add to history
)
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# ─── Environment Setup ────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY") # Keep for now, though Tavily is often recommended
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set")
# if not SERPAPI_API_KEY: # Make search optional if not always needed
#     logger.warning("SERPAPI_API_KEY not set. Web search functionality will be limited.")
#     SERPAPI_API_KEY = None # Allows graceful degradation

BASE_DIR = os.path.dirname(__file__)
# Configure logging at application entry point (e.g., streamlit_ui.py)
# For library-like files, just get the logger
logger = logging.getLogger(__name__)

# ─── Prompt & Knowledge-Base Loading ──────────────────────────────────────────────
def load_prompt(path: str) -> Dict[str, str]:
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        system = "\n".join(raw["system"]) if isinstance(raw["system"], list) else raw.get("system", "")
        user = "\n".join(raw["user_template"]) if isinstance(raw["user_template"], list) else raw.get("user_template", "")
        return {"system": system, "user": user}
    except FileNotFoundError:
        logger.error(f"Prompt file not found at {path}. Using empty prompts.")
        return {"system": "", "user": ""} # Graceful degradation
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from prompt file {path}. Using empty prompts.")
        return {"system": "", "user": ""}


# Adjust paths if your "prompts" and "knowledge_base" are not directly in BASE_DIR
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")

supervisor_prompt = load_prompt(os.path.join(PROMPT_DIR, "supervisor.json"))
agent1_prompt = load_prompt(os.path.join(PROMPT_DIR, "agent_1.json"))
agent2_prompt = load_prompt(os.path.join(PROMPT_DIR, "agent_2.json"))

def load_kb_file(filename: str) -> str:
    try:
        with open(os.path.join(KB_DIR, filename), encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Knowledge base file {filename} not found in {KB_DIR}. Content will be empty.")
        return ""

tools_kb = load_kb_file("LGarchitect_tools_slim.txt")
multi_kb = load_kb_file("LGarchitect_multi_agent_slim.txt")
core_kb = load_kb_file("LGarchitect_LangGraph_core_slim.txt")


# ─── Data Models ─────────────────────────────────────────────────────────────────
class ClientIntake(BaseModel):
    ClientProfile: Dict[str, Any] = {}
    SalesOps: Dict[str, Any] = {}
    Marketing: Dict[str, Any] = {}
    Retention: Dict[str, Any] = {}
    AIReadiness: Dict[str, Any] = {}
    TechStack: Dict[str, Any] = {}
    GoalsTimeline: Dict[str, Any] = {}
    HAF: Dict[str, Any] = {}
    CII: Dict[str, Any] = {}
    ReferenceDocs: Optional[str] = "" # Made optional as per streamlit_ui

class IntakeSummary(BaseModel):
    ClientProfile: Dict[str, Any]
    Good: List[str]
    Bad: List[str]
    Ugly: List[str]
    SolutionSummary: str
    WorkflowOutline: List[str]
    HAF: Dict[str, Any]
    CII: Dict[str, Any]

class ClientFacingReport(BaseModel):
    report_markdown: str

class DevFacingReport(BaseModel):
    blueprint_graph: str

class GraphState(TypedDict):
    session_id: str # Added session_id
    intake: ClientIntake
    websummary: Optional[str] # Made optional
    summary: Optional[IntakeSummary] # Made optional
    client_report: Optional[ClientFacingReport] # Made optional
    dev_report: Optional[DevFacingReport] # Made optional
    error: Optional[Dict[str, Any]] # Made optional

# ─── LLM & Search Tool Initialization ─────────────────────────────────────────────
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, openai_api_key=OPENAI_API_KEY)

# Initialize search_tool only if API key is available
search_tool = None
if SERPAPI_API_KEY:
    try:
        search_tool = SerpAPIWrapper(api_key=SERPAPI_API_KEY)
        logger.info("SerpAPIWrapper initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize SerpAPIWrapper: {e}")
else:
    logger.warning("SERPAPI_API_KEY not found. Web search node will be skipped or limited.")


# ─── Helpers ──────────────────────────────────────────────────────────────────────
def strip_fences(text: str) -> str:
    if not isinstance(text, str): return "" # Handle non-string input
    text = re.sub(r"^```[\w]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```$", "", text, flags=re.MULTILINE)
    return text.strip()

def robust_invoke(messages: List[BaseMessage], max_retries: int = 2, initial_delay: int = 1) -> Any: # Use BaseMessage
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            logger.warning(f"LLM invoke failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(initial_delay * (2 ** attempt)) # Exponential backoff
            else: # Last attempt failed
                logger.error("LLM invoke permanently failed after retries", exc_info=True)
                raise RuntimeError("LLM invoke failed after multiple retries") from e

# ─── Node 1: Supervisor Agent (Now primarily for validation/enrichment within the pipeline) ───
def supervisor_node(state: GraphState) -> Dict[str, Any]:
    try:
        intake_model = state["intake"]
        session_id = state["session_id"]
        logger.info(f"[SUPERVISOR_NODE] Processing for session_id: {session_id}")

        current_intake_snapshot = json.dumps(intake_model.model_dump(), indent=2)
        memory_context_docs = query_vector_memory(
            current_intake_snapshot,
            k=3,
            filter_metadata={"session_id": session_id}
        )
        memory_context_str = "\n".join([doc.page_content for doc in memory_context_docs])

        chat_history_obj = get_chat_history(session_id)
        processed_history = chat_history_obj.messages

        msgs: List[BaseMessage] = [SystemMessage(content=supervisor_prompt["system"])]
        if memory_context_str:
            msgs.append(SystemMessage(content=f"Relevant Context from Memory:\n{memory_context_str}"))
        msgs.extend(processed_history)
        task_instruction = supervisor_prompt["user"].replace("{RAW_INTAKE_JSON}", current_intake_snapshot)
        msgs.append(HumanMessage(content=task_instruction))

        resp = robust_invoke(msgs)
        content = strip_fences(resp.content)

        try:
            parsed_response = json.loads(content)
            validated_intake_data = parsed_response.get("validated_intake", parsed_response.get("updated_fields", intake_model.model_dump()))
            if not isinstance(validated_intake_data, dict):
                logger.warning("[SUPERVISOR_NODE] LLM response for intake was not a dictionary. Using original intake.")
                final_intake_model = intake_model
            else:
                final_intake_model = ClientIntake(**validated_intake_data)
            logger.info(f"[SUPERVISOR_NODE] Intake processed for session {session_id}.")

            # Log this validation to vector memory
            add_to_vector_memory(
                f"Supervisor node validated intake: {json.dumps(final_intake_model.model_dump())}",
                metadata={"session_id": session_id, "type": "supervisor_node_summary"}
            )

            # Optionally, add this message to chat history
            add_lc_message_to_history(session_id, HumanMessage(content="Supervisor node validation complete."))

            return {"intake": final_intake_model}

        except json.JSONDecodeError:
            logger.error(f"[SUPERVISOR_NODE] Failed to parse LLM response as JSON: {content}")
            return {"intake": intake_model, "error": {"node": "supervisor", "message": "Failed to parse LLM JSON output."}}
        except ValidationError as ve:
            logger.error(f"[SUPERVISOR_NODE] Pydantic validation error: {ve}")
            return {"intake": intake_model, "error": {"node": "supervisor", "message": f"Pydantic validation error: {ve}"}}

    except Exception as e:
        logger.error(f"[SUPERVISOR_NODE] Error: {e}", exc_info=True)
        return {"error": {"node": "supervisor", "message": str(e)}, "intake": state.get("intake")}

# ─── Node 2: Web Search Agent ─────────────────────────────────────────────────────
def websearch_node(state: GraphState) -> Dict[str, Any]:
    if not search_tool:
        logger.warning("[WEBSEARCH] Search tool not available. Skipping node.")
        return {"websummary": "Web search not available."}
    try:
        intake_data = state["intake"]
        if not intake_data or not intake_data.ClientProfile: # Check if intake_data and ClientProfile exist
            logger.warning("[WEBSEARCH] ClientProfile is missing in intake. Cannot perform web search.")
            return {"websummary": "Client industry not specified for web search."}

        industry = intake_data.ClientProfile.get("industry", "")
        if not industry:
            logger.info("[WEBSEARCH] Industry not specified. Skipping web search.")
            return {"websummary": "Industry not specified for web search."}

        query = f"Top AI automation best practices in {industry} industry"
        logger.info(f"[WEBSEARCH] Querying: {query}")
        summary = search_tool.run(query)
        logger.info(f"[WEBSEARCH] Summary received for {industry}.")
        return {"websummary": summary or "No information found from web search."}

    except Exception as e:
        logger.error(f"[WEBSEARCH] Error: {e}", exc_info=True)
        return {"error": {"node": "websearch", "message": str(e)}, "websummary": "Error during web search."}


# ─── Node 3: Solution Summarizer Agent ────────────────────────────────────────────
def summarizer_node(state: GraphState) -> Dict[str, Any]:
    try:
        intake_model = state["intake"]
        session_id = state["session_id"]
        logger.info(f"[SUMMARIZER] Processing for session_id: {session_id}")

        raw_json_intake = json.dumps(intake_model.model_dump(), indent=2)
        websum = state.get("websummary", "No web summary available.")

        sys_ctx = f"{agent1_prompt['system']}\n\nWebSummary:\n{websum}\n\nRelevant Knowledge Base Snippet:\n{multi_kb}"
        user_ctx = agent1_prompt["user"].replace("{RAW_INTAKE_JSON}", raw_json_intake)

        msgs: List[BaseMessage] = [
            SystemMessage(content=sys_ctx),
            HumanMessage(content=user_ctx)
        ]

        resp = robust_invoke(msgs)
        content = strip_fences(resp.content)

        try:
            summary_data = json.loads(content)
            summary = IntakeSummary(**summary_data) # Validate with Pydantic
            logger.info(f"[SUMMARIZER] Summary parsed and validated for session {session_id}.")
            # Optionally add summary to vector memory
            # add_to_vector_memory(f"Summarizer output: {json.dumps(summary.model_dump())}",
            #                      metadata={"session_id": session_id, "type": "summarizer_output"})
            return {"summary": summary}
        except json.JSONDecodeError:
            logger.error(f"[SUMMARIZER] Failed to parse LLM summary as JSON: {content}")
            return {"error": {"node": "summarizer", "message": "Failed to parse LLM summary JSON."}, "summary": None}
        except ValidationError as ve:
            logger.error(f"[SUMMARIZER] Pydantic validation error for summary: {ve}. Raw content: {content}")
            return {"error": {"node": "summarizer", "message": f"Summary validation error: {ve}"}, "summary": None}


    except Exception as e:
        logger.error(f"[SUMMARIZER] Unexpected error: {e}", exc_info=True)
        return {"error": {"node": "summarizer", "message": str(e)}, "summary": None}

# ─── Node 4: Report Generation Agent ─────────────────────────────────────────────
def report_node(state: GraphState) -> Dict[str, Any]:
    try:
        summary_model = state.get("summary")
        session_id = state["session_id"]
        logger.info(f"[REPORT_NODE] Processing for session_id: {session_id}")

        if not summary_model:
            logger.warning("[REPORT_NODE] Intake summary is missing. Cannot generate reports.")
            return {
                "error": {"node": "report", "message": "Intake summary missing."},
                "client_report": ClientFacingReport(report_markdown="Error: Intake summary not available to generate client report."),
                "dev_report": DevFacingReport(blueprint_graph="Error: Intake summary not available to generate developer report.")
            }

        summary_json = json.dumps(summary_model.model_dump(), indent=2)
        sys_ctx = f"{agent2_prompt['system']}\n\nRelevant Knowledge Base - Core Concepts:\n{core_kb}\n\nRelevant Knowledge Base - Tools:\n{tools_kb}"
        user_ctx = agent2_prompt["user"].replace("{SUMMARY_JSON}", summary_json)

        msgs: List[BaseMessage] = [
            SystemMessage(content=sys_ctx),
            HumanMessage(content=user_ctx)
        ]
        resp = robust_invoke(msgs)
        content = strip_fences(resp.content)

        try:
            data = json.loads(content)
            client_report_md = data.get("client_report", "Client report could not be generated.")
            dev_report_md = data.get("developer_report", "Developer report could not be generated.")

            logger.info(f"[REPORT_NODE] Reports generated and parsed for session {session_id}.")
            return {
                "client_report": ClientFacingReport(report_markdown=client_report_md),
                "dev_report": DevFacingReport(blueprint_graph=dev_report_md)
            }
        except json.JSONDecodeError:
            logger.error(f"[REPORT_NODE] Failed to parse LLM reports as JSON: {content}")
            return {
                "error": {"node": "report", "message": "Failed to parse LLM reports JSON."},
                "client_report": ClientFacingReport(report_markdown=f"Error: Could not parse report from LLM. Raw: {content}"),
                "dev_report": DevFacingReport(blueprint_graph=f"Error: Could not parse report from LLM. Raw: {content}")
            }

    except Exception as e:
        logger.error(f"[REPORT_NODE] Error: {e}", exc_info=True)
        return {
            "error": {"node": "report", "message": str(e)},
            "client_report": ClientFacingReport(report_markdown=f"Error generating client report: {e}"),
            "dev_report": DevFacingReport(blueprint_graph=f"Error generating dev report: {e}")
        }
# ─── Build LangGraph Pipeline ────────────────────────────────────────────────────
builder = StateGraph(GraphState)
builder.add_node("supervisor_node_pipeline", supervisor_node) # Renamed to avoid conflict if 'supervisor' is a keyword
builder.add_node("websearch", websearch_node)
builder.add_node("summarize", summarizer_node)
builder.add_node("report", report_node)

builder.add_edge(START, "supervisor_node_pipeline")
builder.add_edge("supervisor_node_pipeline", "websearch")
builder.add_edge("websearch", "summarize")
builder.add_edge("summarize", "report")
builder.add_edge("report", END)

graph = builder.compile()
logger.info("LangGraph pipeline compiled.")

# ─── Public API for batch runs ───────────────────────────────────────────────────
def run_pipeline(raw_intake: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    try:
        intake_model = ClientIntake(**raw_intake)
        logger.info(f"Running pipeline for session_id: {session_id} with intake: {intake_model.ClientProfile.get('name')}")

        initial_graph_state = {
            "session_id": session_id,
            "intake": intake_model,
            "websummary": None,
            "summary": None,
            "client_report": None,
            "dev_report": None,
            "error": None
        }
        final_state = graph.invoke(initial_graph_state)

        if final_state.get("error"):
            logger.error(f"Pipeline error for session {session_id}: {final_state['error']}")
            return {"error": final_state["error"]}

        client_report_obj = final_state.get("client_report")
        dev_report_obj = final_state.get("dev_report")

        # Example: Optionally clear chat history after pipeline run (for demonstration)
        clear_chat_history(session_id)

        return {
            "client_report": client_report_obj.report_markdown if client_report_obj else "Client report generation failed or was not run.",
            "dev_report": dev_report_obj.blueprint_graph if dev_report_obj else "Developer report generation failed or was not run.",
        }
    except ValidationError as ve:
        logger.error(f"Pydantic validation error at start of pipeline for session {session_id}: {ve}")
        return {"error": {"node": "initialization", "message": f"Intake validation error: {ve}"}}
    except Exception as e:
        logger.error(f"Unexpected error in run_pipeline for session {session_id}: {e}", exc_info=True)
        return {"error": {"node": "pipeline_execution", "message": str(e)}}


# ─── Supervisor Chat Chain for UI ────────────────────────────────────────────────
def create_supervisor_chain():
    llm_chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, openai_api_key=OPENAI_API_KEY)
    # Ensure supervisor_prompt is loaded correctly
    if not supervisor_prompt["system"] or not supervisor_prompt["user"]:
        logger.error("Supervisor system or user prompt is empty. Chain may not function correctly.")
        # Provide a default minimal prompt if empty to avoid errors with from_template
        default_system = "You are a helpful assistant."
        default_user = "Current Intake State: {RAW_INTAKE_JSON}\n\nUser Query: Please assist." # Modified to just take intake
        current_system_prompt = supervisor_prompt["system"] or default_system
        current_user_prompt = supervisor_prompt["user"] or default_user
    else:
        current_system_prompt = supervisor_prompt["system"]
        current_user_prompt = supervisor_prompt["user"]


    # The ChatPromptTemplate will have system, history (from MessagesPlaceholder), and the human message.
    # The human message comes from HumanMessagePromptTemplate.from_template(current_user_prompt)
    # This template expects {RAW_INTAKE_JSON}. The actual user utterance is part of the `history`.
    prompt_messages = [
        SystemMessagePromptTemplate.from_template(current_system_prompt),
        MessagesPlaceholder(variable_name="history"), # This will take the list of BaseMessage from UI
        HumanMessagePromptTemplate.from_template(current_user_prompt) # This provides the task framing
    ]
    # The `input_variables` for this prompt template are "history" and "RAW_INTAKE_JSON".
    chat_prompt_template = ChatPromptTemplate.from_messages(prompt_messages)
    logger.info("Supervisor chat chain created.")
    return chat_prompt_template | llm_chat

supervisor_chain = create_supervisor_chain()

# ─── CLI/Test Hook ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # More verbose for CLI test
    logger.info("Running graph.py in CLI mode for testing.")
    test_session_id = "cli_test_session_001"
    sample_intake_path = os.path.join(BASE_DIR, "sample_intake.json")
    try:
        with open(sample_intake_path, encoding="utf-8") as f:
            sample = json.load(f)
        logger.info(f"Loaded sample intake from {sample_intake_path}")

        # Simulate UI saving chat history before pipeline run
        # (streamlit_ui.py would use StreamlitChatMessageHistory's messages)
        # For testing, we can manually create some history for this session_id
        # Note: memory_manager.py's get_chat_history() will be called by supervisor_node
        
        # Clear any previous history for this test session
        clear_chat_history(test_session_id)
        
        # Add some dummy history as if it came from the UI chat
        # These need to be BaseMessage objects if supervisor_node expects them via get_chat_history
        from langchain_core.messages import HumanMessage, AIMessage
        initial_history_messages = [
            HumanMessage(content="Hello, I'd like to start the intake."),
            AIMessage(content="Great! What is your name and business name?")
        ]
        # Persist this dummy history using memory_manager's functions
        # This simulates what add_messages_to_history would do if called from streamlit_ui.py
        history_obj = get_chat_history(test_session_id) # Ensure history obj is created
        for msg in initial_history_messages:
            history_obj.add_message(msg)
        # _persist_chat_history(test_session_id) # get_chat_history and add_message ensure persistence if modified to do so

        logger.info(f"Running pipeline with session_id: {test_session_id}")
        reports = run_pipeline(sample, test_session_id)

        if reports.get("error"):
            logger.error(f"Pipeline error: {reports['error']}")
        else:
            logger.info("\n==== CLIENT REPORT ====\n" + reports.get("client_report", "Not generated."))
            logger.info("\n==== DEV REPORT ====\n" + reports.get("dev_report", "Not generated."))

    except FileNotFoundError:
        logger.error(f"Sample intake file not found at {sample_intake_path}. Cannot run CLI test.")
    except Exception as e:
        logger.error(f"Error during CLI test: {e}", exc_info=True)