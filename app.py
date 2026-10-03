import os
import json
import streamlit as st
from groq import Groq, BadRequestError
from tavily import TavilyClient
from dotenv import load_dotenv

# ---------- SETUP ----------
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MODEL = "openai/gpt-oss-120b"


# ---------- WEB SEARCH ----------
def web_search(query):
    results = tavily.search(query=query, max_results=3)

    return "\n\n".join(
        f"Source: {r['url']}\nContent: {r['content']}"
        for r in results["results"]
    )


tools = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current, real-time, or recent information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
}]


# ---------- VERIFICATION ----------
def verify_answer(answer, sources):

    prompt = f"""
You are a silent fact-checker.

Compare the answer with the source material.

Answer:
{answer}

Source material:
{sources}

If the answer is accurate, reply:
CONFIRMED

If it contains unsupported or incorrect information, reply:
CORRECTION: <corrected answer only>
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ---------- MEMORY ----------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        {
            "role": "system",
            "content": "Use the web_search tool for current or real-time facts."
        }
    ]


# ---------- ASK ----------
def ask(question):

    history = st.session_state.conversation_history
    history.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=tools,
            temperature=0
        )
        message = response.choices[0].message

    except BadRequestError as e:
        if "tool_use_failed" not in str(e):
            raise

        # Fallback if model produces an invalid tool call
        results = web_search(question)

        history.append({
            "role": "user",
            "content": (
                f"Web search results:\n\n{results}\n\n"
                "Answer the original question using these results."
            )
        })

        message = None

    # ---------- NO SEARCH ----------
    if message and not message.tool_calls:
        answer = message.content
        history.append({"role": "assistant", "content": answer})
        return answer

    # ---------- SEARCH ----------
    if message:
        history.append({
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
        })

        sources = ""

        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            sources = web_search(args["query"])

            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": sources
            })

    else:
        sources = results

    # ---------- FINAL ANSWER ----------
    response = client.chat.completions.create(
        model=MODEL,
        messages=history,
        tool_choice="none",
        temperature=0
    )

    draft = response.choices[0].message.content

    # ---------- VERIFICATION ----------
    verification = verify_answer(draft, sources)

    if verification.startswith("CORRECTION:"):
        answer = verification.replace("CORRECTION:", "", 1).strip()
    else:
        answer = draft

    history.append({
        "role": "assistant",
        "content": answer
    })

    return answer


# ---------- UI ----------
st.set_page_config(
    page_title="Live AI Assistant",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Live AI Assistant")
st.caption("Search the web, verify answers, and remember our conversation.")

with st.sidebar:
    st.header("About")
    st.write(
        "GPT-OSS-120B + Tavily Search + Streamlit"
    )

    if st.button("Clear conversation"):
        st.session_state.conversation_history = [
            {
                "role": "system",
                "content": "Use the web_search tool for current or real-time facts."
            }
        ]
        st.rerun()


# ---------- CHAT HISTORY ----------
for msg in st.session_state.conversation_history:
    if msg["role"] in ("user", "assistant") and msg.get("content"):
        st.chat_message(msg["role"]).write(msg["content"])


# ---------- CHAT INPUT ----------
if prompt := st.chat_input("Ask me anything..."):

    st.chat_message("user").write(prompt)

    with st.spinner("Thinking..."):
        answer = ask(prompt)

    st.chat_message("assistant").write(answer)