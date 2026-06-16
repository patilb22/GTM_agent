"""
Starter GTM content agent using LangChain + LangGraph.
Install:
  pip install langchain langgraph langchain-openai langchain-community langchain-chroma chromadb pydantic
Run:
  export OPENAI_API_KEY=...
  export LANGSMITH_TRACING=true
  export LANGSMITH_API_KEY=...
  python src/gtm_agent.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Dict, List, TypedDict
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class ContentSuite(BaseModel):
    linkedin_post: str = Field(description="LinkedIn launch post")
    promo_email: str = Field(description="Promotional email with subject line")
    blog_draft: str = Field(description="Short blog draft")
    ad_copy_variations: List[str] = Field(description="3-5 paid ad variants")


class Review(BaseModel):
    score: int = Field(ge=1, le=10)
    issues: List[str]
    revision_notes: str
    approved: bool


class GTMState(TypedDict):
    messages: Annotated[list, add_messages]
    input_brief: str
    retrieved_context: str
    audience: str
    tone: str
    suite: Dict[str, object]
    review: Dict[str, object]
    human_feedback: str


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)
structured_writer = llm.with_structured_output(ContentSuite)
structured_reviewer = llm.with_structured_output(Review)
_retriever = None


ROOT_DIR = Path(__file__).resolve().parents[1]


def build_vectorstore() -> Chroma:
    docs = [
        Document(page_content=(ROOT_DIR / "data" / "products.csv").read_text(encoding="utf-8"), metadata={"source": "products.csv"}),
        Document(page_content=(ROOT_DIR / "data" / "events.md").read_text(encoding="utf-8"), metadata={"source": "events.md"}),
        Document(page_content=(ROOT_DIR / "data" / "past_campaigns.md").read_text(encoding="utf-8"), metadata={"source": "past_campaigns.md"}),
    ]
    return Chroma.from_documents(
        docs,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="gtm_context",
    )


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = build_vectorstore().as_retriever(search_kwargs={"k": 4})
    return _retriever


@tool
def retrieve_gtm_context(query: str) -> str:
    """Retrieve launch dates, product specs, audience notes, offers, CTAs, and past messaging."""
    docs = get_retriever().invoke(query)
    return "\n\n".join(f"SOURCE={d.metadata.get('source')}\n{d.page_content}" for d in docs)


def research_node(state: GTMState) -> GTMState:
    context = retrieve_gtm_context.invoke(state["input_brief"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Identify the likely target audience and best tone from the context. Return concise JSON-ish text."),
        ("human", "Input: {brief}\n\nContext:\n{context}"),
    ])
    result = llm.invoke(prompt.format_messages(brief=state["input_brief"], context=context)).content
    return {
        **state,
        "retrieved_context": context,
        "audience": "PMs, founders, marketing-adjacent roles, consultants",
        "tone": "practical, credible, operator-friendly",
        "messages": state.get("messages", []) + [{"role": "assistant", "content": str(result)}],
    }


def generate_node(state: GTMState) -> GTMState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior product marketing manager. Generate a grounded, ready-to-edit GTM content suite. Do not invent unsupported facts."),
        ("human", "Brief: {brief}\nAudience: {audience}\nTone: {tone}\nSource context:\n{context}"),
    ])
    suite = structured_writer.invoke(prompt.format_messages(
        brief=state["input_brief"], audience=state["audience"], tone=state["tone"], context=state["retrieved_context"]
    ))
    return {**state, "suite": suite.model_dump()}


def review_node(state: GTMState) -> GTMState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Critique the suite for factual grounding, channel fit, consistency, tone alignment, and unsupported claims."),
        ("human", "Source context:\n{context}\n\nSuite:\n{suite}"),
    ])
    review = structured_reviewer.invoke(prompt.format_messages(context=state["retrieved_context"], suite=state["suite"]))
    return {**state, "review": review.model_dump()}


def human_review_node(state: GTMState) -> GTMState:
    feedback = interrupt({
        "task": "Review GTM suite",
        "suite": state["suite"],
        "review_agent_feedback": state["review"],
        "instruction": "Approve, or provide revision notes."
    })
    return {**state, "human_feedback": str(feedback)}


def revise_node(state: GTMState) -> GTMState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Revise the GTM suite using reviewer and human feedback. Preserve factual grounding."),
        ("human", "Context:\n{context}\n\nOriginal suite:\n{suite}\n\nReview:\n{review}\n\nHuman feedback:\n{feedback}"),
    ])
    revised = structured_writer.invoke(prompt.format_messages(
        context=state["retrieved_context"], suite=state["suite"], review=state["review"], feedback=state.get("human_feedback", "")
    ))
    return {**state, "suite": revised.model_dump()}


def route_after_review(state: GTMState) -> str:
    return "human_review" if not state["review"].get("approved", False) else "human_review"


graph = StateGraph(GTMState)
graph.add_node("research", research_node)
graph.add_node("generate", generate_node)
graph.add_node("review", review_node)
graph.add_node("human_review", human_review_node)
graph.add_node("revise", revise_node)
graph.add_edge(START, "research")
graph.add_edge("research", "generate")
graph.add_edge("generate", "review")
graph.add_edge("review", "human_review")
graph.add_edge("human_review", "revise")
graph.add_edge("revise", END)

app = graph.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-launchpilot-001"}}
    first = app.invoke({"input_brief": "Build a content suite for the LaunchPilot AI Product Hunt launch."}, config=config)
    print(first)
    # Resume after interrupt:
    # final = app.invoke(Command(resume="Approved; make the email more founder-focused."), config=config)
    # print(final["suite"])
