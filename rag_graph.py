from __future__ import annotations

from typing import List, TypedDict, Optional

from langchain.schema import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, END


class RAGState(TypedDict, total=False):
    messages: List[BaseMessage]         # historique conversation
    question: str                       # question courante
    docs: List[Document]                # docs récupérés
    answer: str                         # réponse finale
    k: int                              # nb passages
    model_name: str                     # modèle groq
    retriever: object                   # retriever langchain


SYSTEM_PROMPT = (
    "Tu es un assistant de révision. Tu dois répondre UNIQUEMENT à partir du CONTEXTE fourni.\n"
    "Si l'information n'est pas dans le contexte, dis clairement : "
    "\"Je ne trouve pas la réponse dans les documents.\".\n"
    "À la fin, ajoute une section 'Sources' listant (source - page) utilisées.\n"
    "Réponse en français, claire et structurée."
)


def _format_context(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "document")
        page = d.metadata.get("page", "?")
        blocks.append(f"[Source: {src} - page {page}]\n{d.page_content}")
    return "\n\n".join(blocks)


def make_llm(model_name: str) -> ChatGroq:
    # GROQ_API_KEY doit être dans l'env
    return ChatGroq(model=model_name, temperature=0.2)


def retrieve_node(state: RAGState) -> RAGState:
    retriever = state["retriever"]
    question = state["question"]
    docs = retriever.get_relevant_documents(question)
    return {"docs": docs}


def generate_node(state: RAGState) -> RAGState:
    docs = state.get("docs", [])
    question = state["question"]
    model_name = state["model_name"]

    llm = make_llm(model_name)

    context = _format_context(docs)
    user_prompt = f"CONTEXTE:\n{context}\n\nQUESTION:\n{question}\n"

    # On inclut messages passés + system prompt
    messages = state.get("messages", [])
    llm_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    llm_messages.extend(messages)  # mémoire conversationnelle
    llm_messages.append(HumanMessage(content=user_prompt))

    resp = llm.invoke(llm_messages)
    answer = resp.content if hasattr(resp, "content") else str(resp)

    # On met à jour l'historique
    new_messages = messages + [HumanMessage(content=question), AIMessage(content=answer)]
    return {"answer": answer, "messages": new_messages}


def build_rag_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()
