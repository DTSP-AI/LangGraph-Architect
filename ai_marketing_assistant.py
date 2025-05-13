import os
import streamlit as st
from dotenv import load_dotenv
from graph import run_pipeline

# ─── Load environment ───────────────────────────────────────────────────────────
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("🔑 OPENAI_API_KEY not set in environment!")
    st.stop()

# ─── Streamlit UI Setup ─────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Business Optimization Intake", layout="wide")
st.title("🧠 AI Solutions Discovery & Optimization Intake")

# ─── Sidebar: Test Mode Toggle ──────────────────────────────────────────────────
test_mode = st.sidebar.checkbox("🧪 Enable Test Client Mode", value=False)
os.environ["TEST_MODE"] = "true" if test_mode else "false"
if test_mode:
    st.sidebar.success("✅ Test Client Mode is ON")
    st.warning("⚠️ Running in TEST MODE with dynamic GPT-generated client data.")

st.sidebar.markdown("---")

# ─── Sidebar: Business Intake Form ─────────────────────────────────────────────
st.sidebar.header("📋 Business Intake Form")
user_name        = st.sidebar.text_input("Your Name")
business_name    = st.sidebar.text_input("Business Name")
website          = st.sidebar.text_input("Business Website")
industry         = st.sidebar.selectbox("Industry", ["Jewelry","Med Spa","Real Estate","Fitness","Other"])
location         = st.sidebar.text_input("Location")
annual_revenue   = st.sidebar.number_input("Annual Revenue (USD)", min_value=0, step=1000, value=0, format="%d")
employees        = st.sidebar.number_input("Number of Employees", min_value=0, step=1, value=0, format="%d")

# ─── Sidebar: Sales & Marketing ─────────────────────────────────────────────────
st.sidebar.subheader("Sales & Marketing")
sales_process    = st.sidebar.text_area("Describe your current sales process:")
lead_tools       = st.sidebar.text_area("Tools for leads & appointments:")
has_crm          = st.sidebar.selectbox("Do you use a CRM?", ["Yes","No"])
crm_name         = st.sidebar.text_input("If yes, which CRM?")
booking_process  = st.sidebar.text_area("How are appointments booked?")
follow_up        = st.sidebar.text_area("How do you track follow-ups or missed leads?")
channels         = st.sidebar.multiselect("Active Marketing Channels", ["Google Ads","Meta Ads","TikTok","SEO","Influencer","Referral","Events"])
lead_routing     = st.sidebar.text_area("How are leads captured & routed?")
lead_action      = st.sidebar.text_area("What happens after a lead comes in?")
existing_auto    = st.sidebar.text_area("Any automations in place?")

# ─── Sidebar: Retention & AI Readiness ───────────────────────────────────────────
st.sidebar.subheader("Retention & AI Readiness")
sales_cycle      = st.sidebar.slider("Avg Sales Cycle (days)", 1, 180, 30)
follow_up_tacts  = st.sidebar.text_area("Follow-up tactics for missed contacts:")
retention_progs  = st.sidebar.text_area("Loyalty or re-engagement programs:")
uses_ai          = st.sidebar.selectbox("Are you using AI?", ["Yes","No"])
ai_tools         = st.sidebar.text_area("If yes, describe your AI setup:")
manual_areas     = st.sidebar.multiselect("Most manual tasks:", ["Lead follow-up","Appointment setting","Content creation","Customer questions"])
dream_auto       = st.sidebar.text_area("What would you automate if it worked perfectly?")

# ─── Sidebar: Tech & Goals ──────────────────────────────────────────────────────
st.sidebar.subheader("Tech Stack & Goals")
tools            = st.sidebar.multiselect("Current Tools in Use", ["Calendly","Shopify","Squarespace","Twilio","Stripe","Zapier","Klaviyo","Mailchimp","GoHighLevel"])
api_access       = st.sidebar.selectbox("API/Admin access?", ["Yes","No","Not sure"])
comms            = st.sidebar.selectbox("Preferred communication", ["Text","Email","Phone","DMs","Website Chat"])
goals            = st.sidebar.text_area("Top 3 revenue goals (next 6 months):")
biggest_problem  = st.sidebar.text_area("What’s the #1 problem right now?")
comfort          = st.sidebar.selectbox("Comfort with automation/AI:", ["Bring on the robots","Need guidance","Start simple"])
engagement       = st.sidebar.selectbox("Preferred engagement model:", ["Done-For-You","Hybrid","DIY with Support"])
timeline         = st.sidebar.selectbox("Implementation timeline:", ["<30 days","30-60 days","60-90 days","Flexible"])

# ─── Sidebar: HAF & CII ─────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🔧 HAF & CII")
critical_roles   = st.sidebar.text_area("Key team roles:")
role_respons      = st.sidebar.text_area("Responsibilities for each role:")
workflow_map     = st.sidebar.text_area("Sequence: contact → delivery:")
ai_tasks         = st.sidebar.text_area("AI-eligible tasks:")
data_sources     = st.sidebar.text_area("Systems storing data:")
memory_context   = st.sidebar.text_area("Historical context for agents:")
tools_function   = st.sidebar.text_area("Tools by function:")
api_ready        = st.sidebar.text_area("API/admin access to tools?")
compliance       = st.sidebar.text_area("Compliance/regulatory constraints:")
realtime         = st.sidebar.text_area("Workflows needing real-time execution:")
async_flows      = st.sidebar.text_area("Workflows that can run asynchronously:")

# ─── Main: Generate Report ──────────────────────────────────────────────────────
if st.button("🧠 Generate Full Report & Scope"):
    with st.spinner("Processing…"):
        raw_data = {
            "ClientProfile": {
                "name":     user_name,
                "business": business_name,
                "website":  website,
                "industry": industry,
                "location": location,
                "revenue":  annual_revenue,
                "employees":employees
            },
            "SalesOps": {
                "sales_process": sales_process,
                "lead_tools":    lead_tools,
                "crm":           crm_name if has_crm=="Yes" else "None",
                "booking":       booking_process,
                "followups":     follow_up
            },
            "Marketing": {
                "channels":     channels,
                "routing":      lead_routing,
                "post_lead":    lead_action,
                "automations":  existing_auto
            },
            "Retention": {
                "sales_cycle":       sales_cycle,
                "follow_up_tactics": follow_up_tacts,
                "programs":          retention_progs
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
                "CriticalRoles":    critical_roles,
                "KeyWorkflows":     workflow_map,
                "AIEligibleTasks":  ai_tasks
            },
            "CII": {
                "DataSources":        data_sources,
                "MemoryRequirements": memory_context,
                "ToolsRequired":      tools_function,
                "APIReadiness":       api_ready,
                "SecurityNotes":      compliance,
                "Latency": {
                    "Realtime": realtime,
                    "Async":    async_flows
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
