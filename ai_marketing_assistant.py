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

# ─── Streamlit UI Setup ─────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Business Optimization Intake", layout="wide")
st.title("🧠 AI Solutions Discovery & Optimization Intake")

# ─── Sidebar: Test Mode Controls (Top-Left) ────────────────────────────────────
test_mode = st.sidebar.checkbox("🧪 Enable Test Client Mode", value=False)
if test_mode:
    os.environ["TEST_MODE"] = "true"
    st.sidebar.success("✅ Test Client Mode is ON")
    if st.sidebar.button("🔄 Load New Test Client"):
        # This rerun will trigger inject_test_data_node next pipeline run
        st.experimental_rerun()
else:
    os.environ["TEST_MODE"] = "false"

# ─── Sidebar: Settings Header (Optional) ───────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")

# ─── Main: Test Mode Indicator ─────────────────────────────────────────────────
if os.getenv("TEST_MODE", "false") == "true":
    st.warning("⚠️ Running in TEST MODE with dynamic GPT-generated client data.")

# ─── Sidebar: Business Intake Form ─────────────────────────────────────────────
st.sidebar.header("📋 Business Intake Form")
user_name            = st.sidebar.text_input("Your Name")
business_name        = st.sidebar.text_input("Business Name")
website              = st.sidebar.text_input("Business Website")
industry             = st.sidebar.selectbox("Industry", ["Jewelry", "Med Spa", "Real Estate", "Fitness", "Other"])
location             = st.sidebar.text_input("Location")
annual_revenue       = st.sidebar.number_input("Annual Revenue (USD)", min_value=0, step=1000, value=0, format="%d")
employees            = st.sidebar.number_input("Number of Employees", min_value=0, step=1, value=0, format="%d")

# ─── Main Intake Fields ────────────────────────────────────────────────────────
sales_process        = st.sidebar.text_area("Describe your current sales process:")
lead_tools           = st.sidebar.text_area("What tools do you use for leads and appointments?")
has_crm              = st.sidebar.selectbox("Do you use a CRM?", ["Yes", "No"])
crm_name             = st.sidebar.text_input("If yes, which CRM?")
booking_process      = st.sidebar.text_area("How are appointments booked?")
follow_up            = st.sidebar.text_area("How do you track follow-ups or missed leads?")
channels             = st.sidebar.multiselect("Active Marketing Channels", ["Google Ads", "Meta Ads", "TikTok", "SEO", "Influencer", "Referral", "Events"])
lead_routing         = st.sidebar.text_area("How are leads captured and routed?")
lead_action          = st.sidebar.text_area("What happens after a lead comes in?")
existing_automations = st.sidebar.text_area("Any automations in place?")
sales_cycle          = st.sidebar.slider("Average Sales Cycle (days)", 1, 180, 30)
follow_up_tactics    = st.sidebar.text_area("Follow-up tactics for missed or abandoned contacts:")
retention_programs   = st.sidebar.text_area("Loyalty or re-engagement programs:")
uses_ai              = st.sidebar.selectbox("Are you using AI currently?", ["Yes", "No"])
ai_tools             = st.sidebar.text_area("If yes, describe your AI tools/setup:")
manual_areas         = st.sidebar.multiselect("Manual tasks you spend time on:", ["Lead follow-up", "Appointment setting", "Content creation", "Customer questions"])
dream_automation     = st.sidebar.text_area("What would you automate if it worked perfectly?")
tools                = st.sidebar.multiselect("Current Tools in Use", ["Calendly", "Shopify", "Squarespace", "Twilio", "Stripe", "Zapier", "Klaviyo", "Mailchimp", "GoHighLevel"])
api_access           = st.sidebar.selectbox("Do you have API/admin access?", ["Yes", "No", "Not sure"])
comms                = st.sidebar.selectbox("Preferred communication method:", ["Text", "Email", "Phone", "DMs", "Website Chat"])
goals                = st.sidebar.text_area("Top 3 revenue goals (next 6 months):")
biggest_problem      = st.sidebar.text_area("What’s the #1 problem you’re solving right now?")
comfort              = st.sidebar.selectbox("Comfort level with automation/AI:", ["Bring on the robots", "Need guidance", "Start simple"])
engagement           = st.sidebar.selectbox("Preferred engagement model:", ["Done-For-You", "Hybrid", "DIY with Support"])
timeline             = st.sidebar.selectbox("Implementation timeline:", ["<30 days", "30-60 days", "60-90 days", "Flexible"])

# ─── Sidebar: HAF & CII Sections ────────────────────────────────────────────────
st.sidebar.header("🔧 HAF & CII")
critical_roles       = st.sidebar.text_area("Key team roles:")
role_responsibilities= st.sidebar.text_area("Responsibilities for each role:")
workflow_map         = st.sidebar.text_area("Sequence from first contact to fulfillment:")
ai_task_opportunities= st.sidebar.text_area("Where could AI reduce manual work?")
data_sources         = st.sidebar.text_area("Systems storing customer/product data:")
contextual_memory    = st.sidebar.text_area("Historical context useful for agents:")
tools_by_function    = st.sidebar.text_area("Tools by function:")
api_readiness        = st.sidebar.text_area("API/admin access to those tools?")
compliance_flags     = st.sidebar.text_area("Compliance or regulatory constraints:")
realtime_flows       = st.sidebar.text_area("Workflows needing real-time execution:")
batch_or_async_flows = st.sidebar.text_area("Workflows that can run asynchronously:")

# ─── Trigger the Graph Pipeline ─────────────────────────────────────────────────
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
                "channels":   channels,
                "routing":    lead_routing,
                "post_lead":  lead_action,
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
                "CriticalRoles":    critical_roles,
                "KeyWorkflows":     workflow_map,
                "AIEligibleTasks":  ai_task_opportunities
            },
            "CII": {
                "DataSources":        data_sources,
                "MemoryRequirements": contextual_memory,
                "ToolsRequired":      tools_by_function,
                "APIReadiness":       api_readiness,
                "SecurityNotes":      compliance_flags,
                "Latency": {
                    "Realtime": realtime_flows,
                    "Async":    batch_or_async_flows
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
