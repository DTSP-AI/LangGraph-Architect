# ai_marketing_assistant.py

import os
import json
import streamlit as st
from dotenv import load_dotenv
from graph import run_pipeline, supervisor_chain

# ─── Load environment ───────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 OPENAI_API_KEY not set in environment!")
    st.stop()

# ─── Load UI configuration ─────────────────────────────────────────────────────
CONFIG_PATH = os.path.join("config", "ui_options.json")
try:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        ui_opts = json.load(f)
        CHANNEL_OPTIONS = ui_opts.get("channels", [])
        TOOL_OPTIONS    = ui_opts.get("tools", [])
except Exception:
    CHANNEL_OPTIONS = ["Google Ads", "Meta Ads", "TikTok", "SEO", "Influencer", "Referral", "Events"]
    TOOL_OPTIONS    = ["Calendly", "Shopify", "Squarespace", "Twilio", "Stripe", "Zapier", "Klaviyo", "Mailchimp", "GoHighLevel"]

# ─── Streamlit UI Setup ─────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Business Optimization Intake", layout="wide")
st.title("🧠 AI Solutions Discovery & Optimization Intake")

# ─── Sidebar Intake Form ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Business Intake Form")

    # persist each field in session_state
    def text_input_state(key, label):
        if key not in st.session_state:
            st.session_state[key] = ""
        st.session_state[key] = st.text_input(label, value=st.session_state[key])
        return st.session_state[key]

    def number_input_state(key, label, **kwargs):
        if key not in st.session_state:
            st.session_state[key] = kwargs.get("value", 0)
        st.session_state[key] = st.number_input(label, value=st.session_state[key], **{k:v for k,v in kwargs.items() if k!="value"})
        return st.session_state[key]

    def textarea_state(key, label):
        if key not in st.session_state:
            st.session_state[key] = ""
        st.session_state[key] = st.text_area(label, value=st.session_state[key])
        return st.session_state[key]

    def selectbox_state(key, label, options):
        if key not in st.session_state:
            st.session_state[key] = options[0]
        st.session_state[key] = st.selectbox(label, options, index=options.index(st.session_state[key]) if st.session_state[key] in options else 0)
        return st.session_state[key]

    def multiselect_state(key, label, options):
        if key not in st.session_state:
            st.session_state[key] = []
        st.session_state[key] = st.multiselect(label, options, default=st.session_state[key])
        return st.session_state[key]

    def slider_state(key, label, min_value, max_value, value):
        if key not in st.session_state:
            st.session_state[key] = value
        st.session_state[key] = st.slider(label, min_value, max_value, st.session_state[key])
        return st.session_state[key]

    user_name      = text_input_state("user_name", "Your Name")
    business_name  = text_input_state("business_name", "Business Name")
    website        = text_input_state("website", "Business Website")
    industry       = selectbox_state("industry", "Industry", ["Jewelry", "Med Spa", "Real Estate", "Fitness", "Other"])
    location       = text_input_state("location", "Location")
    annual_revenue = number_input_state("annual_revenue", "Annual Revenue (USD)", min_value=0, step=1000, format="%d", value=0)
    employees      = number_input_state("employees", "Number of Employees", min_value=0, step=1, format="%d", value=0)

    sales_process        = textarea_state("sales_process", "Describe your current sales process:")
    lead_tools           = textarea_state("lead_tools", "What tools do you currently use for leads and appointments?")
    has_crm              = selectbox_state("has_crm", "Do you use a CRM?", ["Yes", "No"])
    crm_name             = text_input_state("crm_name", "Which CRM do you use (if any)?")
    booking_process      = textarea_state("booking_process", "How are appointments currently booked?")
    follow_up            = textarea_state("follow_up", "How do you track follow-ups or missed leads?")

    channels             = multiselect_state("channels", "Active Marketing Channels", CHANNEL_OPTIONS)
    lead_routing         = textarea_state("lead_routing", "How are leads captured and routed?")
    lead_action          = textarea_state("lead_action", "Describe what happens after a lead comes in:")
    existing_automations = textarea_state("existing_automations", "Any automations currently in place?")

    sales_cycle          = slider_state("sales_cycle", "Average Sales Cycle (days)", 1, 180, 30)
    follow_up_tactics    = textarea_state("follow_up_tactics", "How do you follow up with missed calls, abandoned carts, or no-shows?")
    retention_programs   = textarea_state("retention_programs", "Any current loyalty, membership, or re-engagement programs?")

    uses_ai              = selectbox_state("uses_ai", "Are you using AI currently?", ["Yes", "No"])
    ai_tools             = textarea_state("ai_tools", "If yes, describe your AI tools or setup:")
    manual_areas         = multiselect_state("manual_areas", "Where do you spend the most manual time?", ["Lead follow-up", "Appointment setting", "Content creation", "Customer questions"])
    dream_auto           = textarea_state("dream_auto", "What would you automate tomorrow if it worked perfectly?")

    tools                = multiselect_state("tools", "Current Tools in Use", TOOL_OPTIONS)
    api_access           = selectbox_state("api_access", "Do you have admin/API access to these tools?", ["Yes", "No", "Not sure"])
    comms                = selectbox_state("comms", "Preferred customer communication method:", ["Text", "Email", "Phone", "DMs", "Website Chat"])

    goals           = textarea_state("goals", "Top 3 revenue goals (next 6 months):")
    biggest_problem = textarea_state("biggest_problem", "What’s the #1 problem you’re trying to solve right now?")
    comfort         = selectbox_state("comfort", "Comfort level with automation/AI:", ["Bring on the robots", "Need guidance", "Start simple"])
    engagement      = selectbox_state("engagement", "Preferred engagement model:", ["Done-For-You", "Hybrid", "DIY with Support"])
    timeline        = selectbox_state("timeline", "Implementation timeline:", ["<30 days", "30-60 days", "60-90 days", "Flexible"])

    with st.expander("🔰 HAF (Hierarchical Agent Framework)"):
        haf_roles     = textarea_state("haf_roles", "List critical roles and responsibilities")
        haf_workflows = textarea_state("haf_workflows", "Map key workflows (e.g., Lead → Sale → Delivery)")
        haf_agents    = textarea_state("haf_agents", "Which tasks could be delegated to AI agents?")

    with st.expander("🧩 CII (Cognitive Infrastructure Intake)"):
        memory_needs    = textarea_state("memory_needs", "What memory or data history do agents need?")
        agent_tools_cfg = textarea_state("agent_tools_cfg", "List specific APIs/tools needed for each agent")
        compliance_flag = textarea_state("compliance_flag", "Any compliance or regulatory constraints?")
        realtime_flows  = textarea_state("realtime_flows", "Which workflows need real-time execution?")
        async_flows     = textarea_state("async_flows", "Which can run in background or off-hours?")

# ─── Generate Reports ────────────────────────────────────────────────────────────
if st.sidebar.button("🧠 Generate Full Report & Scope"):
    progress = st.sidebar.progress(0)
    with st.spinner("Processing…"):
        try:
            progress.progress(20)
            raw_data = {
                "ClientProfile": {
                    "name":      user_name,
                    "business":  business_name,
                    "website":   website,
                    "industry":  industry,
                    "location":  location,
                    "revenue":   annual_revenue,
                    "employees": employees
                },
                "SalesOps": {
                    "sales_process": sales_process,
                    "lead_tools":    lead_tools,
                    "crm":           crm_name if has_crm == "Yes" else "None",
                    "booking":       booking_process,
                    "followups":     follow_up
                },
                "Marketing": {
                    "channels":    channels,
                    "routing":     lead_routing,
                    "post_lead":   lead_action,
                    "automations": existing_automations
                },
                "Retention": {
                    "sales_cycle":       sales_cycle,
                    "follow_up_tactics": follow_up_tactics,
                    "programs":          retention_programs
                },
                "AIReadiness": {
                    "uses_ai":      uses_ai,
                    "tools":        ai_tools,
                    "manual_areas": manual_areas,
                    "dream":        dream_auto
                },
                "TechStack": {
                    "tools":      tools,
                    "api_access": api_access,
                    "comms":      comms
                },
                "GoalsTimeline": {
                    "goals":      goals,
                    "problem":    biggest_problem,
                    "comfort":    comfort,
                    "engagement": engagement,
                    "timeline":   timeline
                },
                "HAF": {
                    "CriticalRoles":   haf_roles,
                    "KeyWorkflows":    haf_workflows,
                    "AIEligibleTasks": haf_agents
                },
                "CII": {
                    "MemoryRequirements": memory_needs,
                    "ToolsRequired":       agent_tools_cfg,
                    "SecurityNotes":       compliance_flag,
                    "Latency": {
                        "Realtime": realtime_flows,
                        "Async":    async_flows
                    }
                },
                "ReferenceDocs": ""
            }
            progress.progress(50)
            # run the full pipeline
            out = run_pipeline(raw_data)
            progress.progress(80)
            # display reports
            st.subheader("📄 Client-Facing Report")
            st.markdown(out["client_report"], unsafe_allow_html=True)
            st.subheader("📋 Dev-Facing Blueprint")
            st.markdown(out["dev_report"], unsafe_allow_html=True)
            progress.progress(100)
        except KeyError as ke:
            st.error(f"⚠️ Missing field: {ke}")
        except ValueError as ve:
            st.error(f"⚠️ Invalid value: {ve}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# ─── Supervisor Chat (pinned) ───────────────────────────────────────────────────
st.markdown("---")
st.header("💬 Supervisor Agent Chat")

if "supervisor_history" not in st.session_state:
    st.session_state.supervisor_history = []

chat_container = st.container()
with chat_container:
    # render past messages
    for msg in st.session_state.supervisor_history:
        st.chat_message(msg["role"]).write(msg["content"])
    # input box
    user_input = st.chat_input("Ask the Supervisor Agent…")
    if user_input:
        st.session_state.supervisor_history.append({"role": "user", "content": user_input})
        resp = supervisor_chain.invoke({
            "history": st.session_state.supervisor_history,
            "user_input": user_input
        })
        assistant_content = resp.content.strip()
        st.session_state.supervisor_history.append({"role": "assistant", "content": assistant_content})
        # on next rerun, chat_container will display updated history
