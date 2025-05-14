# LangGraph-Architect\ai_marketing_assistant.py

import os
import streamlit as st
from dotenv import load_dotenv
from graph import run_pipeline

# ─── Load environment ───────────────────────────────────────────────────────────
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("🔑 OPENAI_API_KEY not set in environment!")
    st.stop()

# ─── Load Kodey.ai docs for ReferenceDocs ───────────────────────────────────────
def load_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

kodey_docs = load_txt("knowledge_base/kodey_agent_build_docs.txt")

# ─── Streamlit UI Setup ─────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Business Optimization Intake", layout="wide")
st.title("🧠 AI Solutions Discovery & Optimization Intake")

# ─── Sidebar Intake Form ────────────────────────────────────────────────────────
st.sidebar.header("📋 Business Intake Form")
user_name      = st.sidebar.text_input("Your Name")
business_name  = st.sidebar.text_input("Business Name")
website        = st.sidebar.text_input("Business Website")
industry       = st.sidebar.selectbox("Industry", ["Jewelry", "Med Spa", "Real Estate", "Fitness", "Other"])
location       = st.sidebar.text_input("Location")
annual_revenue = st.sidebar.number_input("Annual Revenue (USD)", min_value=0, step=1000, value=0, format="%d")
employees      = st.sidebar.number_input("Number of Employees", min_value=0, step=1, value=0, format="%d")

# ─── Main Intake Fields ─────────────────────────────────────────────────────────
sales_process        = st.text_area("Describe your current sales process:")
lead_tools           = st.text_area("What tools do you currently use for leads and appointments?")
has_crm              = st.selectbox("Do you use a CRM?", ["Yes", "No"])
crm_name             = st.text_input("Which CRM do you use (if any)?")
booking_process      = st.text_area("How are appointments currently booked?")
follow_up            = st.text_area("How do you track follow-ups or missed leads?")

channels             = st.multiselect(
    "Active Marketing Channels",
    ["Google Ads", "Meta Ads", "TikTok", "SEO", "Influencer", "Referral", "Events"]
)
lead_routing         = st.text_area("How are leads captured and routed?")
lead_action          = st.text_area("Describe what happens after a lead comes in:")
existing_automations = st.text_area("Any automations currently in place?")

sales_cycle          = st.slider("Average Sales Cycle (days)", 1, 180, 30)
follow_up_tactics    = st.text_area("How do you follow up with missed calls, abandoned carts, or no-shows?")
retention_programs   = st.text_area("Any current loyalty, membership, or re-engagement programs?")

uses_ai              = st.selectbox("Are you using AI currently?", ["Yes", "No"])
ai_tools             = st.text_area("If yes, describe your AI tools or setup.")
manual_areas         = st.multiselect(
    "Where do you spend the most manual time?",
    ["Lead follow-up", "Appointment setting", "Content creation", "Customer questions"]
)
dream_automation     = st.text_area("What would you automate tomorrow if it worked perfectly?")

tools                = st.multiselect(
    "Current Tools in Use",
    ["Calendly", "Shopify", "Squarespace", "Twilio", "Stripe", "Zapier", "Klaviyo", "Mailchimp", "GoHighLevel"]
)
api_access           = st.selectbox("Do you have admin/API access to these tools?", ["Yes", "No", "Not sure"])
comms                = st.selectbox(
    "Preferred customer communication method:",
    ["Text", "Email", "Phone", "DMs", "Website Chat"]
)

goals                = st.text_area("Top 3 revenue goals (next 6 months):")
biggest_problem      = st.text_area("What’s the #1 problem you're trying to solve right now?")
comfort              = st.selectbox("Comfort level with automation/AI:", ["Bring on the robots", "Need guidance", "Start simple"])
engagement           = st.selectbox("Preferred engagement model:", ["Done-For-You", "Hybrid", "DIY with Support"])
timeline             = st.selectbox("Implementation timeline:", ["<30 days", "30-60 days", "60-90 days", "Flexible"])

# ─── HAF & CII Sections ─────────────────────────────────────────────────────────
with st.expander("🔰 HAF (Hierarchical Agent Framework)"):
    haf_roles      = st.text_area("List critical roles and responsibilities")
    haf_workflows  = st.text_area("Map key workflows (e.g., Lead → Sale → Delivery)")
    haf_agents     = st.text_area("Which tasks could be delegated to AI agents?")

with st.expander("🧩 CII (Cognitive Infrastructure Intake)"):
    memory_needs        = st.text_area("What memory or data history do agents need?")
    agent_tools_cfg     = st.text_area("List specific APIs/tools needed for each agent")
    compliance_flags    = st.text_area("Any compliance or regulatory constraints?")
    realtime_flows      = st.text_area("Which workflows need real-time execution?")
    async_flows         = st.text_area("Which can run in background or off-hours?")

# ─── Trigger the Graph Pipeline ─────────────────────────────────────────────────
if st.button("🧠 Generate Full Report & Scope"):
    with st.spinner("Processing…"):
        raw_data = {
            "ClientProfile": {
                "name":       user_name,
                "business":   business_name,
                "website":    website,
                "industry":   industry,
                "location":   location,
                "revenue":    annual_revenue,
                "employees":  employees
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
                "dream":        dream_automation
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
                "SecurityNotes":       compliance_flags,
                "Latency": {
                    "Realtime": realtime_flows,
                    "Async":    async_flows
                }
            },
            "ReferenceDocs": kodey_docs
        }

        try:
            out = run_pipeline(raw_data)
            st.subheader("📄 Client-Facing Report")
            st.markdown(out["client_report"], unsafe_allow_html=True)
            st.subheader("📋 Dev-Facing Blueprint")
            st.markdown(out["dev_report"], unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Failed to generate reports: {e}")
