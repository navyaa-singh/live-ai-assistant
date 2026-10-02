import os
import json
import streamlit as st
from groq import Groq
from groq import BadRequestError
from tavily import TavilyClient
from dotenv import load_dotenv

# ---------- LOAD ENVIRONMENT VARIABLES ----------
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ---------- TOOL: WEB SEARCH ----------
def web_search(query):
    results = tavily.search(query=query, max_results=3)

    formatted = ""

    for r in results["results"]:
        formatted += (
            f"Source: {r['url']}\n"
            f"Content: {r['content']}\n\n"
        )

    return formatted


# ---------- TOOL DEFINITION ----------
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current, real-time, or recent information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ---------- VERIFICATION ----------
def verify_answer(answer, source_material):

    check_prompt = f"""
You are a silent fact-checker. Compare the answer below against the source material.

Answer given:
{answer}

Source material:
{source_material}

Does the answer accurately reflect the source material, with no made-up details?

Reply in EXACTLY one of these two formats, with nothing else added:

CONFIRMED

CORRECTION: <replacement answer only — plain final answer text, no explanations, no mention of sources, no meta-commentary about the checking process>
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": check_prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ---------- MEMORY ----------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


# ---------- ASK FUNCTION ----------
def ask(question):

    if len(st.session_state.conversation_history) == 0:

        st.session_state.conversation_history.append(
            {
                "role": "system",
                "content": (
                    "Use the web_search tool for any question about "
                    "current events or real-time facts."
                )
            }
        )

    st.session_state.conversation_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    tool_call_failed = False

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=st.session_state.conversation_history,
            tools=tools,
            temperature=0
        )

        message = response.choices[0].message

    except BadRequestError as e:

        if "tool_use_failed" in str(e):

            tool_call_failed = True
            message = None

        else:
            raise


    # ---------- SEARCH RESULTS ----------
    search_results_text = None


    # ---------- TOOL CALL FAILED ----------
    if tool_call_failed:

        # The model tried to search but produced a broken call.
        # Force a real search ourselves using the user's raw question.

        search_results_text = web_search(question)

        st.session_state.conversation_history.append(
            {
                "role": "user",
                "content": (
                    f"Here are live web search results for your question:\n\n"
                    f"{search_results_text}\n\n"
                    f"Please answer the original question using only this information."
                )
            }
        )


    # ---------- MODEL REQUESTED WEB SEARCH ----------
    elif message.tool_calls:

        st.session_state.conversation_history.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            }
        )

        for tool_call in message.tool_calls:

            args = json.loads(
                tool_call.function.arguments
            )

            search_results_text = web_search(
                args["query"]
            )

            st.session_state.conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": search_results_text
                }
            )


    # ---------- NO WEB SEARCH REQUIRED ----------
    else:

        st.session_state.conversation_history.append(
            {
                "role": "assistant",
                "content": message.content
            }
        )

        return message.content


    # ---------- FINAL ANSWER ----------
    followup = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=st.session_state.conversation_history,
        tool_choice="none",
        temperature=0
    )

    draft_answer = followup.choices[0].message.content


    # ---------- VERIFICATION ----------
    verification = verify_answer(
        draft_answer,
        search_results_text
    )

    if verification.startswith("CORRECTION"):

        answer = verification.replace(
            "CORRECTION:",
            ""
        ).strip()

    else:

        answer = draft_answer


    # ---------- SAVE ANSWER ----------
    st.session_state.conversation_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return answer


# ---------- UI ----------
st.set_page_config(
    page_title="Live AI Assistant",
    page_icon="🔮",
    layout="centered"
)


# ---------- CUSTOM CSS ----------
st.markdown(
    """
    <style>

        @import url(
            'https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap'
        );

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        .stApp {
            background-color: #1e1329;
        }

        h1 {
            color: #e4d4fb;
            font-weight: 800;
            letter-spacing: -0.5px;
        }

        .subtitle {
            color: #c3aee8;
            font-size: 16px;
            margin-top: -10px;
            margin-bottom: 25px;
        }

        .stChatMessage {
            border-radius: 14px;
            padding: 6px 10px;
            background-color: #2a1c3d;
            border: 1px solid #4a3468;
        }

        .stChatMessage p {
            color: #f1e9fb !important;
        }

        [data-testid="stChatMessageAvatarUser"] {
            background-color: #b388ff;
        }

        [data-testid="stChatMessageAvatarAssistant"] {
            background-color: #7e57c2;
        }

        .stChatInput textarea {
            background-color: #2a1c3d !important;
            border: 1px solid #6a4aa8 !important;
            border-radius: 12px !important;
            color: #f1e9fb !important;
        }

        div[data-testid="stChatInput"] button {
            background-color: #7e57c2 !important;
            border-radius: 10px !important;
        }

        hr {
            border-color: #4a3468 !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #241834;
            border-right: 1px solid #4a3468;
        }

        section[data-testid="stSidebar"] h2 {
            color: #e4d4fb;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] div {
            color: #d8c6f0;
        }

        div.stButton > button {
            background-color: #7e57c2;
            color: white;
            border-radius: 10px;
            border: none;
            width: 100%;
        }

        div.stButton > button:hover {
            background-color: #9575cd;
            color: white;
        }

        footer {
            visibility: hidden;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------- SIDEBAR ----------
with st.sidebar:

    st.markdown("## About")

    st.write(
        "This assistant can search the live web, verify its answers against "
        "real sources, and remember context across your conversation."
    )

    st.markdown(
        "**Tech stack:** GPT-OSS-120B (via Groq) · "
        "Tavily Search · Streamlit"
    )

    st.divider()

    if st.button("Clear conversation"):

        st.session_state.conversation_history = []

        st.rerun()


# ---------- MAIN PAGE ----------
st.title("Live AI Assistant")

st.markdown(
    '<p class="subtitle">'
    'Ask anything — I search the web, verify my answers, and remember our chat.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# ---------- DISPLAY CHAT HISTORY ----------
for msg in st.session_state.conversation_history:

    if (
        isinstance(msg, dict)
        and msg["role"] in ("user", "assistant")
        and msg.get("content")
    ):

        st.chat_message(
            msg["role"]
        ).write(
            msg["content"]
        )


# ---------- CHAT INPUT ----------
if prompt := st.chat_input("Ask me anything..."):

    st.chat_message("user").write(prompt)

    with st.spinner("Thinking..."):

        answer = ask(prompt)

    st.chat_message("assistant").write(answer)