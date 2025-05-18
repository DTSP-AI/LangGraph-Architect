import os
import json
import streamlit as st
from dotenv import load_dotenv

from graph import run_pipeline, supervisor_chain
from memory_manager import clear_chat_history

############################
# 2. Load Environment
############################
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("🔑 OPENAI_API_KEY not set in environment!")
    st.stop()

############################
# 3. Page Setup
############################
st.set_page_config(
    page_title="AI Business Optimization Intake",
    layout="wide",
    initial_sidebar_state="collapsed"
)

############################
# 4. Session State Initialization
############################
def init_session():
    if "intake" not in st.session_state:
        st.session_state.intake = {
            "ClientProfile": {"name": "", "business": "", "website": "", "industry": "", "location": "", "revenue": 0, "employees": 0},
            "SalesOps":       {"sales_process": "", "lead_tools": "", "crm": "", "booking": "", "followups": ""},
            "Marketing":      {"channels": [], "routing": "", "post_lead": "", "automations": ""},
            "Retention":      {"sales_cycle": 30, "follow_up_tactics": "", "programs": ""},
            "AIReadiness":    {"uses_ai": "", "tools": "", "manual_areas": [], "dream": ""},
            "TechStack":      {"tools": [], "api_access": "", "comms": ""},
            "GoalsTimeline":  {"goals": "", "problem": "", "comfort": "", "engagement": "", "timeline": ""},
            "HAF":            {"roles": "", "workflows": "", "agents": ""},
            "CII":            {"memory": "", "tools": "", "compliance": "", "latency": ""}
        }
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_started" not in st.session_state:
        st.session_state.chat_started = False
    if "reports" not in st.session_state:
        st.session_state.reports = {}
    if "intake_complete" not in st.session_state:
        st.session_state.intake_complete = False

init_session()

############################
# 5. Helpers
############################
def update_intake(data: dict):
    for section, fields in data.items():
        if section in st.session_state.intake:
            for key, val in fields.items():
                if key in st.session_state.intake[section]:
                    st.session_state.intake[section][key] = val

def reset_all():
    clear_chat_history(st.session_state.intake["ClientProfile"]["name"])
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session()
    st.experimental_rerun()

############################
# 6. Sidebar: Controls
############################
with st.sidebar:
    if st.button("🔄 Reset All"):
        reset_all()

############################
# 7. Main Panel: Supervisor Chat
############################
st.header("💬 Supervisor Agent Chat")
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Begin intake with the Supervisor Agent...")
if user_input:
    st.session_state.chat_started = True
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    response = supervisor_chain.invoke({
        "history": st.session_state.chat_history,
        "user_input": user_input,
        "intake": st.session_state.intake
    })
    assistant_msg = response.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})

    # extract JSON payload
    try:
        start = assistant_msg.index("{")
        payload = json.loads(assistant_msg[start:])
        update_intake(payload.get("validated_intake", payload))
    except (ValueError, json.JSONDecodeError):
        pass

    st.experimental_rerun()

############################
# 8. Expandable Intake Form
############################
with st.expander("📝 Review/Edit Intake Form", expanded=st.session_state.intake_complete):
    cp = st.session_state.intake["ClientProfile"]
    st.subheader("Client Profile")
    cp["name"]      = st.text_input("Your Name", value=cp["name"])
    cp["business"]  = st.text_input("Business Name", value=cp["business"])
    cp["website"]   = st.text_input("Business Website", value=cp["website"])
    cp["industry"]  = st.selectbox("Industry", ["Jewelry","Med Spa","Real Estate","Fitness","Other"], index=(["Jewelry","Med Spa","Real Estate","Fitness","Other"].index(cp["industry"]) if cp["industry"] in ["Jewelry","Med Spa","Real Estate","Fitness","Other"] else 0))
    cp["location"]  = st.text_input("Location", value=cp["location"])
    cp["revenue"]   = st.number_input("Annual Revenue (USD)", min_value=0, step=1000, value=cp["revenue"], format="%d")
    cp["employees"] = st.number_input("Number of Employees", min_value=0, step=1, value=cp["employees"], format="%d")

    so = st.session_state.intake["SalesOps"]
    st.subheader("Sales & Operations")
    so["sales_process"] = st.text_area("Sales process:", value=so["sales_process"])
    so["lead_tools"]    = st.text_area("Lead & appointment tools:", value=so["lead_tools"])
    so["crm"]           = st.text_input("CRM (if any):", value=so["crm"])
    so["booking"]       = st.text_area("Booking process:", value=so["booking"])
    so["followups"]     = st.text_area("Follow-ups tracking:", value=so["followups"])

    mk = st.session_state.intake["Marketing"]
    st.subheader("Marketing")
    mk["channels"]    = st.multiselect("Marketing Channels", ["Google Ads","Meta Ads","TikTok","SEO","Influencer","Referral","Events"], default=mk["channels"])
    mk["routing"]     = st.text_area("Lead routing:", value=mk["routing"])
    mk["post_lead"]   = st.text_area("Post-lead actions:", value=mk["post_lead"])
    mk["automations"] = st.text_area("Existing automations:", value=mk["automations"])

    rt = st.session_state.intake["Retention"]
    st.subheader("Engagement & Retention")
    rt["sales_cycle"]       = st.slider("Sales cycle (days)", 1, 180, rt["sales_cycle"])
    rt["follow_up_tactics"] = st.text_area("Follow-up tactics:", value=rt["follow_up_tactics"])
    rt["programs"]          = st.text_area("Retention programs:", value=rt["programs"])

    ai = st.session_state.intake["AIReadiness"]
    st.subheader("AI Readiness")
    ai["uses_ai"]      = st.selectbox("Using AI?", ["Yes","No"], index=(0 if ai["uses_ai"]=="Yes" else 1))
    ai["tools"]        = st.text_area("AI tools:", value=ai["tools"])
    ai["manual_areas"] = st.multiselect("Manual time spent:", ["Lead follow-up","Appointment setting","Content creation","Customer questions"], default=ai["manual_areas"])
    ai["dream"]        = st.text_area("Dream automation:", value=ai["dream"])

    ts = st.session_state.intake["TechStack"]
    st.subheader("Tech Stack")
    ts["tools"]      = st.multiselect("Tools in use:", ["Calendly","Shopify","Squarespace","Twilio","Stripe","Zapier","Klaviyo","Mailchimp","GoHighLevel"], default=ts["tools"])
    ts["api_access"] = st.selectbox("Admin/API access?", ["Yes","No","Not sure"], index=(["Yes","No","Not sure"].index(ts["api_access"]) if ts["api_access"] in ["Yes","No","Not sure"] else 0))
    ts["comms"]      = st.selectbox("Comm method:", ["Text","Email","Phone","DMs","Website Chat"], index=(["Text","Email","Phone","DMs","Website Chat"].index(ts["comms"]) if ts["comms"] in ["Text","Email","Phone","DMs","Website Chat"] else 0))

    gt = st.session_state.intake["GoalsTimeline"]
    st.subheader("Goals & Timeline")
    gt["goals"]      = st.text_area("Revenue goals:", value=gt["goals"])
    gt["problem"]    = st.text_area("Key problem:", value=gt["problem"])
    gt["comfort"]    = st.selectbox("Comfort level:", ["Bring on the robots","Need guidance","Start simple"], index=(["Bring on the robots","Need guidance","Start simple"].index(gt["comfort"]) if gt["comfort"] in ["Bring on the robots","Need guidance","Start simple"] else 0))
    gt["engagement"] = st.selectbox("Engagement model:", ["Done-For-You","Hybrid","DIY with Support"], index=(["Done-For-You","Hybrid","DIY with Support"].index(gt["engagement"]) if gt["engagement"] in ["Done-For-You","Hybrid","DIY with Support"] else 0))
    gt["timeline"]   = st.selectbox("Implementation timeline:", ["<30 days","30-60 days","60-90 days","Flexible"], index=(["<30 days","30-60 days","60-90 days","Flexible"].index(gt["timeline"]) if gt["timeline"] in ["<30 days","30-60 days","60-90 days","Flexible"] else 0))

    haf = st.session_state.intake["HAF"]
    st.subheader("HAF: Hierarchical Agent Framework")
    haf["roles"]     = st.text_area("Critical roles:", value=haf["roles"])
    haf["workflows"] = st.text_area("Key workflows:", value=haf["workflows"])
    haf["agents"]    = st.text_area("AI-eligible tasks:", value=haf["agents"])

    cii = st.session_state.intake["CII"]
    st.subheader("CII: Cognitive Infrastructure Intake")
    cii["memory"]     = st.text_area("Memory needs:", value=cii["memory"])
    cii["tools"]      = st.text_area("Agent tools:", value=cii["tools"])
    cii["compliance"] = st.text_area("Compliance notes:", value=cii["compliance"])
    cii["latency"]    = st.text_area("Real-time vs async:", value=cii["latency"])

############################
# 9. Generate Reports
############################
if st.button("🚀 Generate Full Report & Scope", disabled=st.session_state.intake_complete):
    result = run_pipeline(st.session_state.intake)
    if result.get("error"):
        err = result["error"]
        st.error(f"{err['node']} error: {err['message']}")
    else:
        st.session_state.reports = result
        st.session_state.intake_complete = True

############################
# 10. Display Reports
############################
if st.session_state.intake_complete:
    st.subheader("📄 Client-Facing Report")
    st.markdown(st.session_state.reports["client_report"], unsafe_allow_html=True)
    st.subheader("📋 Dev-Facing Blueprint")
    st.markdown(st.session_state.reports["dev_report"], unsafe_allow_html=True)
