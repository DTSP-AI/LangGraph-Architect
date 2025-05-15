# streamlit_ui.py

import os
import streamlit as st
from dotenv import load_dotenv
from graph import run_pipeline, supervisor_chain

# ─── Load environment variables ─────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 OPENAI_API_KEY not set in environment!")
    st.stop()

# ─── Streamlit page configuration ────────────────────────────────────────────────
st.set_page_config(page_title="AI Business Optimization Intake", layout="wide")
st.title("🧠 AI Solutions Discovery & Optimization Intake")

# ─── Initialize namespaced intake data in session_state ──────────────────────────
if "intake_data" not in st.session_state:
    st.session_state.intake_data = {
        "ClientProfile": {
            "name": "",
            "business": "",
            "website": "",
            "industry": "",
            "location": "",
            "revenue": 0,
            "employees": 0
        },
        "SalesOps": {
            "sales_process": "",
            "lead_tools": "",
            "crm": "",
            "booking": "",
            "followups": ""
        },
        "Marketing": {
            "channels": [],
            "routing": "",
            "post_lead": "",
            "automations": ""
        },
        "Retention": {
            "sales_cycle": 30,
            "follow_up_tactics": "",
            "programs": ""
        },
        "AIReadiness": {
            "uses_ai": "",
            "tools": "",
            "manual_areas": [],
            "dream": ""
        },
        "TechStack": {
            "tools": [],
            "api_access": "",
            "comms": ""
        },
        "GoalsTimeline": {
            "goals": "",
            "problem": "",
            "comfort": "",
            "engagement": "",
            "timeline": ""
        },
        "HAF": {
            "CriticalRoles": "",
            "KeyWorkflows": "",
            "AIEligibleTasks": ""
        },
        "CII": {
            "MemoryRequirements": "",
            "ToolsRequired": "",
            "SecurityNotes": "",
            "Latency": {
                "Realtime": "",
                "Async": ""
            }
        },
        "ReferenceDocs": ""
    }

if "intake_complete" not in st.session_state:
    st.session_state.intake_complete = False

if "reports" not in st.session_state:
    st.session_state.reports = {}

if "supervisor_history" not in st.session_state:
    st.session_state.supervisor_history = []

# ─── Sidebar: Mirrored Intake Form ────────────────────────────────────────────────
if not st.session_state.intake_complete:
    with st.sidebar:
        st.header("📋 Business Intake Form")

        cp = st.session_state.intake_data["ClientProfile"]
        cp["name"] = st.text_input("Your Name", value=cp["name"])
        cp["business"] = st.text_input("Business Name", value=cp["business"])
        cp["website"] = st.text_input("Business Website", value=cp["website"])
        industries = ["Jewelry","Med Spa","Real Estate","Fitness","Other"]
        cp["industry"] = st.selectbox(
            "Industry", industries,
            index=industries.index(cp["industry"]) if cp["industry"] in industries else 0
        )
        cp["location"] = st.text_input("Location", value=cp["location"])
        cp["revenue"] = st.number_input(
            "Annual Revenue (USD)", min_value=0, step=1000, format="%d", value=cp["revenue"]
        )
        cp["employees"] = st.number_input(
            "Number of Employees", min_value=0, step=1, format="%d", value=cp["employees"]
        )

        so = st.session_state.intake_data["SalesOps"]
        so["sales_process"] = st.text_area("Describe your current sales process:", value=so["sales_process"])
        so["lead_tools"] = st.text_area("Tools for leads & appointments:", value=so["lead_tools"])
        so["crm"] = st.text_input("Which CRM do you use?", value=so["crm"])
        so["booking"] = st.text_area("How are appointments booked?", value=so["booking"])
        so["followups"] = st.text_area("How do you track follow-ups/missed leads?", value=so["followups"])

        mk = st.session_state.intake_data["Marketing"]
        channels = ["Google Ads","Meta Ads","TikTok","SEO","Influencer","Referral","Events"]
        mk["channels"] = st.multiselect("Active Marketing Channels", channels, default=mk["channels"])
        mk["routing"] = st.text_area("How are leads captured/routed?", value=mk["routing"])
        mk["post_lead"] = st.text_area("What happens after a lead comes in?", value=mk["post_lead"])
        mk["automations"] = st.text_area("Any automations currently in place?", value=mk["automations"])

        rt = st.session_state.intake_data["Retention"]
        rt["sales_cycle"] = st.slider("Average Sales Cycle (days)", 1, 180, value=rt["sales_cycle"])
        rt["follow_up_tactics"] = st.text_area("Follow-up tactics:", value=rt["follow_up_tactics"])
        rt["programs"] = st.text_area("Loyalty/membership programs:", value=rt["programs"])

        ar = st.session_state.intake_data["AIReadiness"]
        uses_ai_opts = ["Yes","No"]
        ar["uses_ai"] = st.selectbox("Are you using AI?", uses_ai_opts,
                                     index=uses_ai_opts.index(ar["uses_ai"]) if ar["uses_ai"] in uses_ai_opts else 1)
        ar["tools"] = st.text_area("If yes, describe your AI tools:", value=ar["tools"])
        manual_opts = ["Lead follow-up","Appointment setting","Content creation","Customer questions"]
        ar["manual_areas"] = st.multiselect("Where most manual time spent?", manual_opts, default=ar["manual_areas"])
        ar["dream"] = st.text_area("Dream automation (perfect world):", value=ar["dream"])

        ts = st.session_state.intake_data["TechStack"]
        tool_opts = ["Calendly","Shopify","Squarespace","Twilio","Stripe","Zapier","Klaviyo","Mailchimp","GoHighLevel"]
        ts["tools"] = st.multiselect("Current Tools in Use", tool_opts, default=ts["tools"])
        api_opts = ["Yes","No","Not sure"]
        ts["api_access"] = st.selectbox("Admin/API access to these tools?", api_opts,
                                        index=api_opts.index(ts["api_access"]) if ts["api_access"] in api_opts else 2)
        comms_opts = ["Text","Email","Phone","DMs","Website Chat"]
        ts["comms"] = st.selectbox("Preferred communication:", comms_opts,
                                   index=comms_opts.index(ts["comms"]) if ts["comms"] in comms_opts else 0)

        gt = st.session_state.intake_data["GoalsTimeline"]
        gt["goals"] = st.text_area("Top 3 revenue goals (6 mo):", value=gt["goals"])
        gt["problem"] = st.text_area("#1 problem to solve:", value=gt["problem"])
        comfort_opts = ["Bring on the robots","Need guidance","Start simple"]
        gt["comfort"] = st.selectbox("Comfort with automation/AI:", comfort_opts,
                                     index=comfort_opts.index(gt["comfort"]) if gt["comfort"] in comfort_opts else 2)
        engagement_opts = ["Done-For-You","Hybrid","DIY with Support"]
        gt["engagement"] = st.selectbox("Preferred engagement model:", engagement_opts,
                                        index=engagement_opts.index(gt["engagement"]) if gt["engagement"] in engagement_opts else 0)
        timeline_opts = ["<30 days","30-60 days","60-90 days","Flexible"]
        gt["timeline"] = st.selectbox("Implementation timeline:", timeline_opts,
                                      index=timeline_opts.index(gt["timeline"]) if gt["timeline"] in timeline_opts else 0)

        haf = st.session_state.intake_data["HAF"]
        with st.expander("🔰 HAF (Hierarchical Agent Framework)"):
            haf["CriticalRoles"] = st.text_area("Critical roles & responsibilities:", value=haf["CriticalRoles"])
            haf["KeyWorkflows"]    = st.text_area("Key workflows map:", value=haf["KeyWorkflows"])
            haf["AIEligibleTasks"] = st.text_area("Tasks for AI agents:", value=haf["AIEligibleTasks"])

        cii = st.session_state.intake_data["CII"]
        with st.expander("🧩 CII (Cognitive Infrastructure Intake)"):
            cii["MemoryRequirements"] = st.text_area("Memory/data needs:", value=cii["MemoryRequirements"])
            cii["ToolsRequired"]      = st.text_area("APIs/tools needed:", value=cii["ToolsRequired"])
            cii["SecurityNotes"]      = st.text_area("Compliance constraints:", value=cii["SecurityNotes"])
            lat = cii["Latency"]
            lat["Realtime"] = st.text_area("Real-time workflows:", value=lat["Realtime"])
            lat["Async"]    = st.text_area("Background/async workflows:", value=lat["Async"])

        # ── Enable Generate only when required fields are populated ──────────────────
        required_keys = [
            ("ClientProfile","name"), ("ClientProfile","business"), ("ClientProfile","website"),
            ("ClientProfile","industry"), ("ClientProfile","location"),
            ("SalesOps","sales_process"), ("SalesOps","lead_tools"), ("SalesOps","crm"),
            ("GoalsTimeline","goals"), ("GoalsTimeline","problem")
        ]
        ready = True
        for section, key in required_keys:
            val = st.session_state.intake_data[section][key]
            if val in [None, "", [], 0]:
                ready = False
                break

        if st.sidebar.button("🧠 Generate Full Report & Scope", disabled=not ready):
            raw_data = st.session_state.intake_data.copy()
            result   = run_pipeline(raw_data)
            if isinstance(result, dict) and "error" in result:
                err = result["error"]
                if isinstance(err, list):
                    for e in err:
                        st.error(f"❌ Pipeline error in node '{e.get('node')}': {e.get('message')}")
                else:
                    st.error(f"❌ Pipeline error in node '{err.get('node')}': {err.get('message')}")
            else:
                st.session_state.intake_complete = True
                st.session_state.reports = result

# ─── Main area: separator + Supervisor Chat + Reports ───────────────────────────
st.markdown("---")
st.header("💬 Supervisor Agent Chat")

# render chat history
for msg in st.session_state.supervisor_history:
    st.chat_message(msg["role"]).write(msg["content"])

# handle new chat input
user_input = st.chat_input("Ask the Supervisor Agent…")
if user_input:
    st.session_state.supervisor_history.append({"role": "user", "content": user_input})
    resp = supervisor_chain.invoke({
        "history":    st.session_state.supervisor_history,
        "user_input": user_input
    })
    assistant_content = resp.content.strip()
    st.session_state.supervisor_history.append({"role": "assistant", "content": assistant_content})
    st.experimental_rerun()

# once intake is complete, show reports
if st.session_state.intake_complete:
    out = st.session_state.reports
    st.subheader("📄 Client-Facing Report")
    st.markdown(out["client_report"], unsafe_allow_html=True)
    st.subheader("📋 Dev-Facing Blueprint")
    st.markdown(out["dev_report"], unsafe_allow_html=True)
