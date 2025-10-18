# primary_gpt_teacher_app_streamlit.py
import os
import time
import numpy as np
import streamlit as st

from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ----------------------------
# Config / Keys
# ----------------------------
GROQ_API_KEY = os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY")
DB_FAISS_PATH = "db_faiss"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------------------
# System Prompt with Teaching Rules
# ----------------------------
STRICT_TEACHING_RULES = """
You are currently STUDYING, and you've asked me to follow these strict rules during this chat.

STRICT RULES:
- Be an approachable, dynamic teacher who helps the user learn by guiding them.
- If you don't know their goals or grade, ask lightly first.
- Build on what they know and connect new ideas to prior knowledge.
- Guide, don't just give answers. Use small steps so the user discovers answers.
- Offer explanations with **context, examples, and clarity**, but keep them concise enough for an active back-and-forth.
- After hard parts, confirm understanding with a short review or question.
- Vary rhythm: mix explanations, small guiding questions, and checks.
- DO NOT DO THE USER'S WORK. Never just give full answers. If asked for a direct solution, respond with a guiding question first.

ALLOWED:
- Teach new concepts with examples, then review with a quick check.
- Homework help by scaffolding thinking, **not giving full solutions**.
- Practice quizzes (one Q at a time) and let user try before revealing.

IMPORTANT:
If user uploads a math or logic problem, do NOT solve immediately. Ask step-by-step guiding questions.
""".strip()

PROMPT = ChatPromptTemplate.from_messages([
    ("system", STRICT_TEACHING_RULES),
    (
        "human",
        (
            "Here is the recent conversation for context:\n{history}\n\n"
            "Additional study resources:\n{context}\n\n"
            "Student: {question}\n\n"
            "Remember: Be a friendly teacher, give useful explanations with examples, ask one guiding question at a time."
        ),
    ),
])

# ----------------------------
# ChatBot Class
# ----------------------------
class ChatBot:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
        self.db = None
        try:
            self.db = FAISS.load_local(DB_FAISS_PATH, self.embeddings, allow_dangerous_deserialization=True)
        except:
            print("[WARN] No FAISS DB found; retrieval will be limited.")

        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="gemma2-9b-it",
            streaming=False,
            temperature=0.2,
        )

        self.doc_texts, self.bm25 = [], None
        self._build_bm25_index()

        self.chain = PROMPT | self.llm | StrOutputParser()

    def _build_bm25_index(self):
        if self.db:
            try:
                all_docs = list(getattr(self.db.docstore, "_dict", {}).values())
                self.doc_texts = [d.page_content for d in all_docs]
                if self.doc_texts:
                    self.bm25 = BM25Okapi([t.split() for t in self.doc_texts])
            except:
                pass

    def hybrid_retrieve_and_rerank(self, query, top_k=5):
        if not self.db or not self.doc_texts:
            return ""
        try:
            docs = self.db.as_retriever(search_kwargs={"k": 20}).invoke(query)
            return "\n\n---\n\n".join([d.page_content for d in docs[:top_k]])
        except:
            return ""

    def generate_answer(self, query: str, history: list) -> str:
        last_turns = history[-4:] if history else []
        formatted_history = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in last_turns])
        context = self.hybrid_retrieve_and_rerank(query)
        try:
            return self.chain.invoke({"history": formatted_history, "context": context, "question": query}).strip()
        except Exception as e:
            return f"Error generating response: {e}"


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Primary GPT – Student Companion", layout="centered", page_icon="🎓")

st.title("🎓 Primary GPT – Student Companion Assistant")
st.caption("A guided learning chatbot that teaches by **asking, explaining, and reviewing** — not just giving answers.")

# Persistent bot + chat history
if "bot" not in st.session_state:
    st.session_state.bot = ChatBot()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for msg in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["assistant"])

# User input
if prompt := st.chat_input("Ask a question or start studying..."):
    st.session_state.messages.append({"user": prompt, "assistant": ""})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response_text = ""
        bot = st.session_state.bot
        full_reply = bot.generate_answer(prompt, [(m["user"], m["assistant"]) for m in st.session_state.messages[:-1]])

        # Typing animation
        for ch in full_reply:
            response_text += ch
            message_placeholder.markdown(response_text + "▌")
            time.sleep(0.01)
        message_placeholder.markdown(response_text)

    st.session_state.messages[-1]["assistant"] = response_text

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

