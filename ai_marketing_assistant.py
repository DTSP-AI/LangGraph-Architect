import os
import streamlit as st
from dotenv import load_dotenv
from graph import run_pipeline
import random
import json

# ─── Load environment ───────────────────────────────────────────────────────────
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("🔑 OPENAI_API_KEY not set in environment!")
    st.stop()

# ─── Dynamic Test Data Toggle ───────────────────────────────────────────────────
test_mode = st.sidebar.checkbox("🧪 Enable Dynamic Test Client Mode", value=False)
refresh_data = False
if test_mode:
    if st.sidebar.button("🔄 New Random Client"):
        refresh_data = True

    with open("test_data_samples.json", "r", encoding="utf-8") as f:
        test_clients = json.load(f)
    selected_test_client = random.choice(test_clients)

# ─── Streamlit UI Setup ─────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Business Optimization Intake", layout="wide")
st.title("🧠 AI Solutions Discovery & Optimization Intake")

# ─── Input Assignment ───────────────────────────────────────────────────────────
def get_value(field):
    return selected_test_client.get(field, "") if test_mode else ""

def get_list(field):
    return selected_test_client.get(field, []) if test_mode else []

def get_number(field):
    return selected_test_client.get(field, 0) if test_mode else 0

# ─── Sidebar Intake Form ────────────────────────────────────────────────────────
st.sidebar.header("📋 Business Intake Form")
user_name = st.sidebar.text_input("Your Name", get_value("name"))
business_name = st.sidebar.text_input("Business Name", get_value("business"))
website = st.sidebar.text_input("Business Website", get_value("website"))
industry = st.sidebar.selectbox("Industry", ["Jewelry", "Med Spa", "Real Estate", "Fitness", "Other"], index=2 if test_mode else 0)
location = st.sidebar.text_input("Location", get_value("location"))
annual_revenue = st.sidebar.number_input("Annual Revenue (USD)", min_value=0, step=1000, value=int(get_number("revenue")), format="%d")
employees = st.sidebar.number_input("Number of Employees", min_value=0, step=1, value=int(get_number("employees")), format="%d")

# ─── Main Intake Fields ─────────────────────────────────────────────────────────
sales_process = st.text_area("Describe your current sales process:", get_value("sales_process"))
lead_tools = st.text_area("What tools do you currently use for leads and appointments?", get_value("lead_tools"))
has_crm = st.selectbox("Do you use a CRM?", ["Yes", "No"], index=0 if get_value("crm") else 1)
crm_name = st.text_input("Which CRM do you use (if any)?", get_value("crm"))
booking_process = st.text_area("How are appointments currently booked?", get_value("booking"))
follow_up = st.text_area("How do you track follow-ups or missed leads?", get_value("followups"))
channels = st.multiselect("Active Marketing Channels", ["Google Ads", "Meta Ads", "TikTok", "SEO", "Influencer", "Referral", "Events"], default=get_list("channels"))
lead_routing = st.text_area("How are leads captured and routed?", get_value("routing"))
lead_action = st.text_area("Describe what happens after a lead comes in:", get_value("post_lead"))
existing_automations = st.text_area("Any automations currently in place?", get_value("automations"))
sales_cycle = st.slider("Average Sales Cycle (days)", 1, 180, int(get_number("sales_cycle")))
follow_up_tactics = st.text_area("How do you follow up with missed calls, abandoned carts, or no-shows?", get_value("follow_up_tactics"))
retention_programs = st.text_area("Any current loyalty, membership, or re-engagement programs?", get_value("programs"))
uses_ai = st.selectbox("Are you using AI currently?", ["Yes", "No"], index=0 if get_value("uses_ai") else 1)
ai_tools = st.text_area("If yes, describe your AI tools or setup.", get_value("tools"))
manual_areas = st.multiselect("Where do you spend the most manual time?", ["Lead follow-up", "Appointment setting", "Content creation", "Customer questions"], default=get_list("manual_areas"))
dream_automation = st.text_area("What would you automate tomorrow if it worked perfectly?", get_value("dream"))
tools = st.multiselect("Current Tools in Use", ["Calendly", "Shopify", "Squarespace", "Twilio", "Stripe", "Zapier", "Klaviyo", "Mailchimp", "GoHighLevel"], default=get_list("tools"))
api_access = st.selectbox("Do you have admin/API access to these tools?", ["Yes", "No", "Not sure"], index=0 if get_value("api_access") == "Yes" else 2)
comms = st.selectbox("Preferred customer communication method:", ["Text", "Email", "Phone", "DMs", "Website Chat"], index=0)
goals = st.text_area("Top 3 revenue goals (next 6 months):", get_value("goals"))
biggest_problem = st.text_area("What’s the #1 problem you're trying to solve right now?", get_value("problem"))
comfort = st.selectbox("Comfort level with automation/AI:", ["Bring on the robots", "Need guidance", "Start simple"], index=0)
engagement = st.selectbox("Preferred engagement model:", ["Done-For-You", "Hybrid", "DIY with Support"], index=0)
timeline = st.selectbox("Implementation timeline:", ["<30 days", "30-60 days", "60-90 days", "Flexible"], index=2)

# ─── HAF & CII Sections ─────────────────────────────────────────────────────────
critical_roles = st.text_area("Who are the key team members or roles in your operations?", get_value("CriticalRoles"))
workflow_map = st.text_area("Describe the sequence from first contact to fulfillment:", get_value("KeyWorkflows"))
ai_task_opportunities = st.text_area("Where could AI reduce manual work?", get_value("AIEligibleTasks"))
data_sources = st.text_area("What systems store your customer/product data?", get_value("DataSources"))
contextual_memory = st.text_area("What historical context would be useful for your agents?", get_value("MemoryRequirements"))
tools_by_function = st.text_area("For each function, what tools do you use?", get_value("ToolsRequired"))
api_readiness = st.text_area("Do you have API/admin access to those tools?", get_value("APIReadiness"))
compliance_flags = st.text_area("Any compliance or regulatory constraints?", get_value("SecurityNotes"))
realtime_flows = st.text_area("Which workflows need real-time execution?", get_value("Realtime"))
batch_or_async_flows = st.text_area("Which can run in background or off-hours?", get_value("Async"))

# ─── Trigger the Graph Pipeline ─────────────────────────────────────────────────
if st.button("🧠 Generate Full Report & Scope"):
    with st.spinner("Processing…"):
        raw_data = {
            "ClientProfile": {
                "name": user_name,
                "business": business_name,
                "website": website,
                "industry": industry,
                "location": location,
                "revenue": annual_revenue,
                "employees": employees
            },
            "SalesOps": {
                "sales_process": sales_process,
                "lead_tools": lead_tools,
                "crm": crm_name if has_crm == "Yes" else "None",
                "booking": booking_process,
                "followups": follow_up
            },
            "Marketing": {
                "channels": channels,
                "routing": lead_routing,
                "post_lead": lead_action,
                "automations": existing_automations
            },
            "Retention": {
                "sales_cycle": sales_cycle,
                "follow_up_tactics": follow_up_tactics,
                "programs": retention_programs
            },
            "AIReadiness": {
                "uses_ai": uses_ai,
                "tools": ai_tools,
                "manual_areas": manual_areas,
                "dream": dream_automation
            },
            "TechStack": {
                "tools": tools,
                "api_access": api_access,
                "comms": comms
            },
            "GoalsTimeline": {
                "goals": goals,
                "problem": biggest_problem,
                "comfort": comfort,
                "engagement": engagement,
                "timeline": timeline
            },
            "HAF": {
                "CriticalRoles": critical_roles.split("\n"),
                "KeyWorkflows": workflow_map.split("\n"),
                "AIEligibleTasks": ai_task_opportunities.split("\n")
            },
            "CII": {
                "MemoryRequirements": contextual_memory.split("\n"),
                "ToolsRequired": tools_by_function.split("\n"),
                "SecurityNotes": compliance_flags.split("\n"),
                "Latency": {
                    "Realtime": realtime_flows.split("\n"),
                    "Async": batch_or_async_flows.split("\n")
                }
            },
            "ReferenceDocs": ""
        }

        try:
            out = run_pipeline(raw_data)
            st.subheader("📄 Client-Facing Report")
            st.markdown(out["client_report"], unsafe_allow_html=True)
            st.subheader("📋 Dev-Facing Blueprint")
            st.markdown(out["dev_report"], unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Failed to generate reports: {e}")
