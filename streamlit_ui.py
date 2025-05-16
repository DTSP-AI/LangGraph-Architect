# streamlit_ui.py

import os
import streamlit as st
from dotenv import load_dotenv
from graph import run_pipeline, supervisor_chain
from memory_manager import clear_chat_history

# ─── Load environment variables ────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("🔑 OPENAI_API_KEY not set in environment!")
    st.stop()

# ─── Streamlit page config ─────────────────────────────────────────
st.set_page_config(page_title="AI Business Optimization Intake", layout="wide")

# ─── Initialize session state ──────────────────────────────────────
if "intake" not in st.session_state:
    st.session_state.intake = {
        "ClientProfile": {"name": "", "business": "", "website": "", "industry": "", "location": ""},
        "SalesOps":       {"sales_process": "", "lead_tools": "", "crm": "", "booking": "", "followups": ""},
        "Marketing":      {"channels": [], "routing": "", "post_lead": "", "automations": []},
        "Retention":      {"sales_cycle": 30, "follow_up_tactics": "", "programs": ""},
        "AIReadiness":    {"uses_ai": "", "tools": "", "manual_areas": [], "dream": ""},
        "TechStack":      {"tools": [], "api_access": "", "comms": ""},
        "GoalsTimeline":  {"goals": "", "problem": "", "comfort": "", "engagement": "", "timeline": ""},
        "HAF":            {"CriticalRoles": "", "KeyWorkflows": "", "AIEligibleTasks": ""},
        "CII":            {"MemoryRequirements": "", "ToolsRequired": "", "SecurityNotes": "", "Latency": {"Realtime": "", "Async": ""}},
        "ReferenceDocs":  ""
    }
if "intake_complete" not in st.session_state:
    st.session_state.intake_complete = False
if "reports" not in st.session_state:
    st.session_state.reports = {}

# ─── Utility to reset everything ────────────────────────────────────
def reset_intake():
    st.session_state.intake_complete = False
    st.session_state.reports = {}
    st.session_state.intake = st.session_state.intake.copy()  # reinit with defaults
    clear_chat_history(st.session_state.intake["ClientProfile"].get("name", "default"))

# ─── Sidebar: mirrored intake form ─────────────────────────────────
with st.sidebar:
    st.header("📋 Business Intake Form")
    if not st.session_state.intake_complete:
        cp = st.session_state.intake["ClientProfile"]
        cp["name"]     = st.text_input("Your Name", value=cp["name"])
        cp["business"] = st.text_input("Business Name", value=cp["business"])
        cp["website"]  = st.text_input("Business Website", value=cp["website"])
        cp["industry"] = st.selectbox("Industry", ["Jewelry","Med Spa","Real Estate","Fitness","Other"], index=(["Jewelry","Med Spa","Real Estate","Fitness","Other"].index(cp["industry"]) if cp["industry"] else 0))
        cp["location"] = st.text_input("Location", value=cp["location"])

        so = st.session_state.intake["SalesOps"]
        so["sales_process"] = st.text_area("Describe your current sales process:", value=so["sales_process"])
        so["lead_tools"]    = st.text_area("What tools do you currently use for leads and appointments?", value=so["lead_tools"])
        so["crm"]           = st.text_input("Which CRM do you use (if any)?", value=so["crm"])
        so["booking"]       = st.text_area("How are appointments currently booked?", value=so["booking"])
        so["followups"]     = st.text_area("How do you track follow-ups or missed leads?", value=so["followups"])

        mk = st.session_state.intake["Marketing"]
        mk["channels"]      = st.multiselect("Active Marketing Channels", ["Google Ads","Meta Ads","TikTok","SEO","Influencer","Referral","Events"], default=mk["channels"])
        mk["routing"]       = st.text_area("How are leads captured and routed?", value=mk["routing"])
        mk["post_lead"]     = st.text_area("Describe what happens after a lead comes in:", value=mk["post_lead"])
        mk["automations"]   = st.text_area("Any automations currently in place?", value=",".join(mk["automations"])).split(",")

        rt = st.session_state.intake["Retention"]
        rt["sales_cycle"]       = st.slider("Average Sales Cycle (days)", 1, 180, rt["sales_cycle"])
        rt["follow_up_tactics"] = st.text_area("How do you follow up with missed calls...?", value=rt["follow_up_tactics"])
        rt["programs"]          = st.text_area("Any loyalty/membership/re-engagement programs?", value=rt["programs"])

        ai = st.session_state.intake["AIReadiness"]
        ai["uses_ai"]      = st.selectbox("Are you using AI currently?", ["Yes","No"], index=(0 if ai["uses_ai"]=="Yes" else 1))
        ai["tools"]        = st.text_area("If yes, describe your AI tools or setup:", value=ai["tools"])
        ai["manual_areas"] = st.multiselect("Where do you spend the most manual time?", ["Lead follow-up","Appointment setting","Content creation","Customer questions"], default=ai["manual_areas"])
        ai["dream"]        = st.text_area("What would you automate tomorrow if it worked perfectly?", value=ai["dream"])

        ts = st.session_state.intake["TechStack"]
        ts["tools"]      = st.multiselect("Current Tools in Use", ["Calendly","Shopify","Squarespace","Twilio","Stripe","Zapier","Klaviyo","Mailchimp","GoHighLevel"], default=ts["tools"])
        ts["api_access"] = st.selectbox("Do you have admin/API access?", ["Yes","No","Not sure"], index=(["Yes","No","Not sure"].index(ts["api_access"]) if ts["api_access"] else 0))
        ts["comms"]      = st.selectbox("Preferred customer communication method:", ["Text","Email","Phone","DMs","Website Chat"], index=(["Text","Email","Phone","DMs","Website Chat"].index(ts["comms"]) if ts["comms"] else 0))

        gt = st.session_state.intake["GoalsTimeline"]
        gt["goals"]      = st.text_area("Top 3 revenue goals:", value=gt["goals"])
        gt["problem"]    = st.text_area("What’s the #1 problem you’re trying to solve?", value=gt["problem"])
        gt["comfort"]    = st.selectbox("Comfort level with automation/AI:", ["Bring on the robots","Need guidance","Start simple"], index=(["Bring on the robots","Need guidance","Start simple"].index(gt["comfort"]) if gt["comfort"] else 0))
        gt["engagement"] = st.selectbox("Preferred engagement model:", ["Done-For-You","Hybrid","DIY with Support"], index=(["Done-For-You","Hybrid","DIY with Support"].index(gt["engagement"]) if gt["engagement"] else 0))
        gt["timeline"]   = st.selectbox("Implementation timeline:", ["<30 days","30-60 days","60-90 days","Flexible"], index=(["<30 days","30-60 days","60-90 days","Flexible"].index(gt["timeline"]) if gt["timeline"] else 0))

        with st.expander("🔰 HAF (Hierarchical Agent Framework)"):
            haf = st.session_state.intake["HAF"]
            haf["CriticalRoles"]   = st.text_area("List critical roles and responsibilities", value=haf["CriticalRoles"])
            haf["KeyWorkflows"]    = st.text_area("Map key workflows", value=haf["KeyWorkflows"])
            haf["AIEligibleTasks"] = st.text_area("Which tasks could be delegated to AI agents?", value=haf["AIEligibleTasks"])

        with st.expander("🧩 CII (Cognitive Infrastructure Intake)"):
            cii = st.session_state.intake["CII"]
            cii["MemoryRequirements"] = st.text_area("What memory or data history do agents need?", value=cii["MemoryRequirements"])
            cii["ToolsRequired"]       = st.text_area("List specific APIs/tools needed", value=cii["ToolsRequired"])
            cii["SecurityNotes"]       = st.text_area("Any compliance or regulatory constraints?", value=cii["SecurityNotes"])
            latency = cii["Latency"]
            latency["Realtime"] = st.text_area("Which workflows need real-time execution?", value=latency["Realtime"])
            latency["Async"]    = st.text_area("Which can run in background/off-hours?", value=latency["Async"])
            cii["Latency"] = latency

        st.text("")  # spacer
        # Check required fields
        required = [
            st.session_state.intake["ClientProfile"]["name"],
            st.session_state.intake["ClientProfile"]["business"],
            st.session_state.intake["ClientProfile"]["website"],
            st.session_state.intake["ClientProfile"]["industry"],
            st.session_state.intake["ClientProfile"]["location"]
        ]
        all_filled = all(required)

        if st.button("🧠 Generate Full Report & Scope", disabled=not all_filled):
            result = run_pipeline(st.session_state.intake)
            if result.get("error"):
                st.sidebar.error(f"❌ {result['error']['node']} error: {result['error']['message']}")
            else:
                st.session_state.reports = result
                st.session_state.intake_complete = True

        if st.button("🔄 Reset Intake"):
            reset_intake()

# ─── Main pane: Reports & Supervisor Chat ────────────────────────────────
if st.session_state.intake_complete:
    st.subheader("📄 Client-Facing Report")
    st.markdown(st.session_state.reports["client_report"], unsafe_allow_html=True)
    st.subheader("📋 Dev-Facing Blueprint")
    st.markdown(st.session_state.reports["dev_report"], unsafe_allow_html=True)

st.markdown("---")
st.header("💬 Supervisor Agent Chat")

if "supervisor_history" not in st.session_state:
    st.session_state.supervisor_history = []

for msg in st.session_state.supervisor_history:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask the Supervisor Agent…")
if user_input:
    st.session_state.supervisor_history.append({"role": "user", "content": user_input})
    response = supervisor_chain.invoke({
        "history": st.session_state.supervisor_history,
        "user_input": user_input
    })
    assistant_content = response.content.strip()
    st.session_state.supervisor_history.append({"role": "assistant", "content": assistant_content})
    st.experimental_rerun()
