# LinkedIn Post Generator with self-review loop (Iterative) UI using streamLit

import os
import re
import textwrap 
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*uses fixed sampling defaults.*",
    category=UserWarning,
)

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv


load_dotenv()

#tools 

search_tool = TavilySearch(max_results = 3)


tools = [search_tool]

#llms 

#writer
writer_llm = ChatGoogleGenerativeAI(model= "gemini-3.5-flash-lite",temperature=0.7)
writer_llm_with_tools = writer_llm.bind_tools(tools)

#reviewer

reviewer_llm = ChatGoogleGenerativeAI(model= "gemini-3.5-flash-lite", temperature=0)

#state building 

class State(TypedDict):
    topic : str 
    messages : Annotated[list,add_messages]
    draft : str 
    review_feedback : str
    is_approved : bool 
    attempt : int


# helpers

def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def clean_post(content) -> str:
    """Extracts plain text and tidies spacing and line breaks."""
    text = extract_text(content).replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def print_post(content, width: int = 80) -> None:
    """Prints the post in a framed, wrapped layout."""
    border = "=" * width
    print(border)
    print("FINAL LINKEDIN POST".center(width))
    print(border)
    print()
    for paragraph in clean_post(content).split("\n\n"):
        flat = " ".join(paragraph.split())
        print(textwrap.fill(flat, width=width))
        print()
    print(border)


#nodes 

WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150–200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)


def writer_node(state : State) -> dict:
    """Writes (or rewrites) the LinkedIn post. Can call Tavily to search first."""
    attempt = state.get("attempt",0) + 1 
    topic = state["topic"]
    previous_feedback = state['review_feedback']

    if attempt == 1:
        user_message = (
            f"Write a LinkedIn post on this topic {topic}"
            f"if you need current info search the web first "
        )
    else:
        user_message = (
            f"your previous draft on '{topic}' was rejected"
            f"Here is the reviewer's feedback \n\n {previous_feedback}\n\n"
            f"Write a new, improved draft that fixes every issue mentiond"
            f"do not repeat the same mistake"
        )
    messages = [("system",WRITER_SYSTEM_PROMPT),("human",user_message)]
    response = writer_llm_with_tools.invoke(messages)

    draft = extract_text(response.content).strip()

    return {
        "messages" : [("human",user_message),response],
        "attempt" : attempt
    }

tool_node = ToolNode(tools)

def extract_draft_node(state:State) -> dict:
    """After the writer finishes tool calls, pulls the final text out as the draft."""
    last_message = state['messages'][-1]
    draft = last_message.content 
    print(f"\n\n generated post \n {draft} \n ")
    return {"draft" : draft}
    

REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim — uses short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone (not corporate-robotic)\n"
    "7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing."
)

def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)

def reviewer_node(state:State) -> dict:
    """Reviews the draft and decides: approve or reject with feedback."""
    draft = state['draft']

    prompt = (
        f"review this LinkedIn post draft : \n"
        f"{draft}\n"
        f"give your reviews"
    )
    response = reviewer_llm.invoke(
        [("system",REVIEWER_SYSTEM_PROMPT),("human",prompt)]
    )
    review_text = extract_text(response.content).strip()
    
    is_approved = "APPROVED" in review_text.upper().split("FEEDBACK")[0]

    if "FEEDBACK:" in review_text:
        feedback = review_text.split("FEEDBACK:", 1)[1].strip()
    else:
        feedback = review_text

    verdict = "APPROVED" if is_approved else "REJECTED"
    print(f"[Verdict: {verdict}]")
    print(f"[Feedback: {feedback}]")

    return {
        "review_feedback": feedback,
        "is_approved": is_approved,
    }

#router function 

def should_use_tool(state:State):
    last_message = state['messages'][-1]

    if getattr(last_message,'tool_calls',None):
        return "tools"
    return "extract_draft"

def should_stop_looping(state:State):
    if state['is_approved']:
        print("post haas been approved \n")
        return END
    if state['attempt'] >= 3:
        print("reached max attempts")
        return END 
    return "writer"

#build the graph 
graph = StateGraph(State)

graph.add_node("writer",writer_node)
graph.add_node("tools",tool_node)
graph.add_node("extract_draft",extract_draft_node)
graph.add_node("reviewer",reviewer_node)

graph.add_edge(START,"writer")

graph.add_conditional_edges(
    "writer",should_use_tool,
)

graph.add_edge("tools","reviewer")
graph.add_edge("extract_draft", "reviewer")

graph.add_conditional_edges(
    "reviewer",should_stop_looping
)

app = graph.compile()


# =====================================================================
# STREAMLIT UI LAYER
# =====================================================================

import html
import streamlit as st

st.set_page_config(
    page_title="LinkedIn Post Studio • Self-Review AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for LinkedIn-inspired executive look & feel
st.markdown(
    """
    <style>
    /* Global spacing & typography */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0A66C2 0%, #004182 100%);
        color: #ffffff;
        padding: 1.75rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(10, 102, 194, 0.25);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        opacity: 0.9;
        margin: 0;
        font-weight: 400;
    }

    /* LinkedIn Feed Post Card */
    .linkedin-card {
        background: #ffffff;
        color: #191919;
        border: 1px solid #e0dfdc;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .li-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1rem;
    }
    .li-avatar {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0A66C2, #0077B5);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.15rem;
        flex-shrink: 0;
    }
    .li-author-name {
        font-weight: 600;
        font-size: 0.95rem;
        color: #191919;
        line-height: 1.2;
    }
    .li-author-sub {
        font-size: 0.78rem;
        color: #666666;
        line-height: 1.3;
    }
    .li-body {
        font-size: 0.95rem;
        line-height: 1.55;
        color: #191919;
        white-space: pre-line;
        margin-bottom: 1.1rem;
    }
    .li-divider {
        border-top: 1px solid #ebebeb;
        margin: 0.75rem 0 0.5rem 0;
    }
    .li-actions {
        display: flex;
        justify-content: space-around;
        color: #5e5e5e;
        font-size: 0.85rem;
        font-weight: 600;
        padding-top: 0.25rem;
    }

    /* Status Badges */
    .badge-approved {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        border: 1px solid #86efac;
    }
    .badge-rejected {
        display: inline-block;
        background: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        border: 1px solid #fca5a5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""

# Sidebar: Agent Specs, Rubric, and History
with st.sidebar:
    st.markdown("### ⚙️ Agent Architecture")
    st.caption("LangGraph Iterative Self-Review Loop")

    google_key_ok = bool(os.getenv("GOOGLE_API_KEY"))
    tavily_key_ok = bool(os.getenv("TAVILY_API_KEY"))

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.markdown(
            f"**Gemini API:** {'🟢 Ready' if google_key_ok else '🔴 Missing'}"
        )
    with col_k2:
        st.markdown(
            f"**Tavily API:** {'🟢 Ready' if tavily_key_ok else '🔴 Missing'}"
        )

    st.divider()
    st.markdown("### 📋 Reviewer Rubric")
    st.markdown(
        """
        1. **Strong Hook** in line 1
        2. **1 Clear Takeaway**
        3. **Skimmable** short paragraphs
        4. **150–200 Words** target length
        5. **Engaging Question / CTA** ending
        6. **Human Tone** (non-robotic)
        7. **Zero Hashtags**
        """
    )

    st.divider()
    st.markdown("### 🕒 Session History")
    if not st.session_state.history:
        st.caption("Generated posts will appear here.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history)):
            status_icon = "✅" if item["is_approved"] else "⚠️"
            if st.button(
                f"{status_icon} {item['topic'][:28]}... (Att: {item['attempt']})",
                key=f"hist_{idx}",
                use_container_width=True,
            ):
                st.session_state.current_result = item

# Hero Header
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">💼 LinkedIn Post Generator Studio</div>
        <div class="hero-subtitle">
            Autonomous Writer + Strict Reviewer Loop powered by LangGraph, Gemini 3.5 Flash-Lite &amp; Tavily Search
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main Two-Column Layout
left_col, right_col = st.columns([1.05, 1.15], gap="large")

with left_col:
    st.subheader("1. Define Your Post Topic")

    # Quick Prompt Starters
    st.caption("Quick topic ideas:")
    preset_cols = st.columns(3)
    presets = [
        "AI Agents in Software Engineering 2026",
        "Why Remote Teams Fail at Async Communication",
        "The Hidden Cost of Microservices",
    ]
    for i, preset in enumerate(presets):
        if preset_cols[i].button(preset, key=f"preset_{i}", use_container_width=True):
            st.session_state.topic_input = preset

    topic = st.text_area(
        "What topic do you want a LinkedIn post about?",
        value=st.session_state.topic_input,
        height=110,
        placeholder="e.g., How small engineering teams are shipping 3x faster with LangGraph agents...",
    ).strip()

    generate_clicked = st.button(
        "🚀 Generate & Self-Review Post",
        type="primary",
        use_container_width=True,
    )

    if generate_clicked:
        if not topic:
            st.warning("Please enter a topic before generating.")
        else:
            initial_state = {
                "topic": topic,
                "messages": [],
                "draft": "",
                "review_feedback": "",
                "is_approved": False,
                "attempt": 0,
            }

            iteration_logs = []
            running_state = dict(initial_state)
            current_iter = {"attempt": 0, "used_search": False, "draft": "", "feedback": "", "is_approved": False}

            st.subheader("2. Live Self-Review Execution Trace")
            with st.status("Running LangGraph Writer ↔ Reviewer loop...", expanded=True) as status_box:
                for step_event in app.stream(initial_state):
                    for node_name, node_output in step_event.items():
                        running_state.update(node_output)

                        if node_name == "writer":
                            att = node_output.get("attempt", running_state.get("attempt", 1))
                            current_iter = {
                                "attempt": att,
                                "used_search": False,
                                "draft": "",
                                "feedback": "",
                                "is_approved": False,
                            }
                            st.write(f"✍️ **Writer Node** — Drafting Attempt #{att}...")

                        elif node_name == "tools":
                            current_iter["used_search"] = True
                            st.write("🌐 **Tavily Search Node** — Fetched live web context.")

                        elif node_name == "extract_draft":
                            cleaned = clean_post(node_output.get("draft", ""))
                            current_iter["draft"] = cleaned
                            st.write(f"📄 **Draft Extracted** ({len(cleaned.split())} words).")

                        elif node_name == "reviewer":
                            approved = node_output.get("is_approved", False)
                            feedback = node_output.get("review_feedback", "")
                            current_iter["is_approved"] = approved
                            current_iter["feedback"] = feedback
                            iteration_logs.append(dict(current_iter))

                            if approved:
                                st.success(f"✅ **Reviewer Verdict (Attempt #{current_iter['attempt']}): APPROVED**")
                            else:
                                st.warning(
                                    f"🔄 **Reviewer Verdict (Attempt #{current_iter['attempt']}): REJECTED**\n\n"
                                    f"*Feedback:* {feedback}"
                                )

                final_status_label = (
                    "✅ Post Approved & Publish-Ready!"
                    if running_state["is_approved"]
                    else f"⚠️ Reached Max Attempts ({running_state['attempt']}/3)"
                )
                status_box.update(label=final_status_label, state="complete", expanded=False)

            result_record = {
                "topic": topic,
                "draft": clean_post(running_state.get("draft", "")),
                "raw_draft": running_state.get("draft", ""),
                "attempt": running_state.get("attempt", 1),
                "is_approved": running_state.get("is_approved", False),
                "review_feedback": running_state.get("review_feedback", ""),
                "iterations": iteration_logs,
            }
            st.session_state.current_result = result_record
            st.session_state.history.append(result_record)

    # Show detailed iteration breakdown if a result is loaded
    if st.session_state.current_result and st.session_state.current_result.get("iterations"):
        st.markdown("#### 🔍 Iteration History")
        for item in st.session_state.current_result["iterations"]:
            badge = "APPROVED ✅" if item["is_approved"] else "REJECTED ❌"
            with st.expander(
                f"Attempt #{item['attempt']} — {badge} {'(Used Web Search 🌐)' if item['used_search'] else ''}",
                expanded=not item["is_approved"],
            ):
                if item["draft"]:
                    st.markdown("**Draft Text:**")
                    st.info(item["draft"])
                else:
                    st.caption("_No draft extracted in this step (tool call routed directly to reviewer)._")
                st.markdown("**Reviewer Feedback:**")
                st.write(item["feedback"])

with right_col:
    st.subheader("3. Final LinkedIn Post Preview")

    res = st.session_state.current_result
    if not res:
        st.info(
            "👈 Enter a topic on the left and click **Generate & Self-Review Post** to preview your final post here."
        )
    else:
        cleaned_text = res["draft"]
        word_count = len(cleaned_text.split()) if cleaned_text else 0
        char_count = len(cleaned_text)

        # KPI Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Status", "Approved" if res["is_approved"] else "Max Retries")
        m2.metric("Attempts", f"{res['attempt']} / 3")
        m3.metric("Word Count", f"{word_count}")
        m4.metric("Characters", f"{char_count}")

        tab_preview, tab_copy, tab_cli = st.tabs(
            ["📱 LinkedIn Feed View", "📋 One-Click Copy", "🖥️ Framed CLI Output"]
        )

        with tab_preview:
            badge_html = (
                '<span class="badge-approved">✓ PUBLISH READY</span>'
                if res["is_approved"]
                else '<span class="badge-rejected">! NEEDS MANUAL EDIT</span>'
            )
            safe_body = html.escape(cleaned_text or "No draft text produced.")

            st.markdown(
                f"""
                <div style="margin-bottom: 0.6rem; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.85rem; color: #666;">Topic: <b>{html.escape(res['topic'])}</b></span>
                    {badge_html}
                </div>
                <div class="linkedin-card">
                    <div class="li-header">
                        <div class="li-avatar">IN</div>
                        <div>
                            <div class="li-author-name">Your LinkedIn Profile • 1st</div>
                            <div class="li-author-sub">Thought Leader &amp; Builder • Content Studio</div>
                            <div class="li-author-sub">Just now • 🌐</div>
                        </div>
                    </div>
                    <div class="li-body">{safe_body}</div>
                    <div class="li-divider"></div>
                    <div class="li-actions">
                        <span>👍 Like</span>
                        <span>💬 Comment</span>
                        <span>🔁 Repost</span>
                        <span>✈️ Send</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if res.get("review_feedback"):
                st.markdown("##### Latest Reviewer Verdict Note")
                if res["is_approved"]:
                    st.success(res["review_feedback"])
                else:
                    st.warning(res["review_feedback"])

        with tab_copy:
            st.caption("Click the copy icon in the top-right corner of the box below:")
            st.code(cleaned_text, language=None)
            st.download_button(
                "⬇️ Download Post (.txt)",
                data=cleaned_text,
                file_name="linkedin_post.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with tab_cli:
            # Renders your original print_post() framed layout inside the UI
            width = 80
            border = "=" * width
            lines = [border, "FINAL LINKEDIN POST".center(width), border, ""]
            for paragraph in clean_post(res["raw_draft"]).split("\n\n"):
                flat = " ".join(paragraph.split())
                lines.append(textwrap.fill(flat, width=width))
                lines.append("")
            lines.append(border)
            st.code("\n".join(lines), language=None)