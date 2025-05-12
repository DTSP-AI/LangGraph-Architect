# File: C:\AI_src\marketing-assistant\ai_marketing_assistant.py

import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# ─── Load environment ───────────────────────────────────────────────────────────
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("🔑 OPENAI_API_KEY not set in environment!")
    st.stop()

# ─── Load Prompts and Kodey Docs ────────────────────────────────────────────────
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_txt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

agent1_prompt = load_json("prompts/agent_1.json")
agent2_prompt = load_json("prompts/agent_2.json")
kodey_docs = load_txt("knowledge_base/kodey_agent_build_docs.txt")

# ─── LLM & UI Init ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Business Optimization Intake", layout="wide")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.9, openai_api_key=openai_key)

# ─── Intake Form ────────────────────────────────────────────────────────────────
st.sidebar.header("📋 Business Intake Form")
user_name = st.sidebar.text_input("Your Name")
business_name = st.sidebar.text_input("Business Name")
website = st.sidebar.text_input("Business Website")
industry = st.sidebar.selectbox("Industry", ["Jewelry", "Med Spa", "Real Estate", "Fitness", "Other"])
location = st.sidebar.text_input("Location")
annual_revenue = st.sidebar.number_input("Annual Revenue (USD)", min_value=0, step=1000, value=0, format="%d")
employees = st.sidebar.number_input("Number of Employees", min_value=0, step=1, value=0, format="%d")

st.title("🧠 AI Solutions Discovery & Optimization Intake")

# ── Main Intake Fields ──────────────────────────────────────────────────────────
sales_process = st.text_area("Describe your current sales process:")
lead_tools = st.text_area("What tools do you currently use for leads and appointments?")
has_crm = st.selectbox("Do you use a CRM?", ["Yes", "No"])
crm_name = st.text_input("Which CRM do you use (if any)?")
booking_process = st.text_area("How are appointments currently booked?")
follow_up = st.text_area("How do you track follow-ups or missed leads?")

channels = st.multiselect("Active Marketing Channels", ["Google Ads","Meta Ads","TikTok","SEO","Influencer","Referral","Events"])
lead_routing = st.text_area("How are leads captured and routed?")
lead_action = st.text_area("Describe what happens after a lead comes in:")
existing_automations = st.text_area("Any automations currently in place?")

sales_cycle = st.slider("Average Sales Cycle (days)", 1, 180, 30)
follow_up_tactics = st.text_area("How do you follow up with missed calls, abandoned carts, or no-shows?")
retention = st.text_area("Any current loyalty, membership, or re-engagement programs?")

uses_ai = st.selectbox("Are you using AI currently?", ["Yes", "No"])
ai_tools = st.text_area("If yes, describe your AI tools or setup.")
manual_areas = st.multiselect("Where do you spend the most manual time?", ["Lead follow-up","Appointment setting","Content creation","Customer questions"])
dream_automation = st.text_area("What would you automate tomorrow if it worked perfectly?")

tools = st.multiselect("Current Tools in Use", ["Calendly","Shopify","Squarespace","Twilio","Stripe","Zapier","Klaviyo","Mailchimp","GoHighLevel"])
api_access = st.selectbox("Do you have admin/API access to these tools?", ["Yes","No","Not sure"])
comms = st.selectbox("Preferred customer communication method:", ["Text","Email","Phone","DMs","Website Chat"])

goals = st.text_area("Top 3 revenue goals (next 6 months):")
biggest_problem = st.text_area("What’s the #1 problem you're trying to solve right now?")
comfort = st.selectbox("Comfort level with automation/AI:", ["Bring on the robots","Need guidance","Start simple"])
engagement = st.selectbox("Preferred engagement model:", ["Done-For-You","Hybrid","DIY with Support"])
timeline = st.selectbox("Implementation timeline:", ["<30 days","30-60 days","60-90 days","Flexible"])

# ─── HAF Inputs ─────────────────────────────────────────────────────────────────
critical_roles = st.text_area("Who are the key team members or roles in your operations (e.g., Sales Rep, Ops Manager, Admin)?")
role_responsibilities = st.text_area("What are the primary responsibilities for each of those roles?")
workflow_map = st.text_area("Describe the sequence of events from first contact to customer delivery or fulfillment:")
ai_task_opportunities = st.text_area("Where do you see opportunities to reduce manual work or delegate tasks to AI?")
handoff_points = st.text_area("Where do tasks typically get passed from one role/team to another?")
decision_points = st.text_area("Which decisions are made by humans today that could potentially be automated?")

# ─── CII Inputs ─────────────────────────────────────────────────────────────────
data_sources = st.text_area("What systems or databases currently store your customer, product, or operational data?")
contextual_memory = st.text_area("What kind of historical context (e.g., past orders, lead history, interaction logs) would be useful for your agents?")
tools_by_function = st.text_area("For each function (e.g., scheduling, payments, marketing), what tools or platforms do you currently use?")
api_readiness = st.text_area("Do you have API access or admin control for the tools you mentioned?")
compliance_flags = st.text_area("Are there any compliance frameworks or certifications you must meet (e.g., HIPAA, GDPR)?")
realtime_flows = st.text_area("Which parts of your workflow need to happen instantly or in real time?")
batch_or_async_flows = st.text_area("Which tasks can happen in the background or after-hours (e.g., reporting, email follow-up)?")

# ─── Trigger Agent Flow ─────────────────────────────────────────────────────────
if st.button("🧠 Generate Full Report & Scope"):
    with st.spinner("Processing..."):

        raw_data = {
            "ClientProfile": {
                "name": user_name, "business": business_name, "website": website,
                "industry": industry, "location": location,
                "revenue": annual_revenue, "employees": employees
            },
            "SalesOps": {
                "sales_process": sales_process, "lead_tools": lead_tools,
                "crm": crm_name if has_crm == "Yes" else "None",
                "booking": booking_process, "followups": follow_up
            },
            "Marketing": {
                "channels": channels, "routing": lead_routing,
                "post_lead": lead_action, "automations": existing_automations
            },
            "Retention": {
                "sales_cycle": sales_cycle, "follow_up_tactics": follow_up_tactics,
                "programs": retention
            },
            "AIReadiness": {
                "uses_ai": uses_ai, "tools": ai_tools,
                "manual_areas": manual_areas, "dream": dream_automation
            },
            "TechStack": {
                "tools": tools, "api_access": api_access, "comms": comms
            },
            "GoalsTimeline": {
                "goals": goals, "problem": biggest_problem,
                "comfort": comfort, "engagement": engagement, "timeline": timeline
            },
            "HAF": {
                "CriticalRoles": critical_roles,
                "KeyWorkflows": workflow_map,
                "AIEligibleTasks": ai_task_opportunities
            },
            "CII": {
                "MemoryRequirements": contextual_memory,
                "ToolsRequired": tools_by_function,
                "SecurityNotes": compliance_flags,
                "Latency": {
                    "Realtime": realtime_flows,
                    "Async": batch_or_async_flows
                }
            },
            "ReferenceDocs": kodey_docs
        }

        # ── AGENT 1 ─ Intake Summary
        agent1_messages = [
            SystemMessage(content=agent1_prompt["system"]),
            HumanMessage(content=agent1_prompt["user_template"].replace("{RAW_INTAKE_JSON}", json.dumps(raw_data, indent=2)))
        ]
        summary_resp = llm(agent1_messages)
        summary = summary_resp.content

        st.subheader("📄 Summary Report")
        st.code(summary, language="json")

        # ── AGENT 2 ─ Final Output
        agent2_messages = [
            SystemMessage(content=agent2_prompt["system"]),
            HumanMessage(content=summary)
        ]
        final_resp = llm(agent2_messages)

        st.subheader("📋 Client & Dev Facing Report")
        st.markdown(final_resp.content)


