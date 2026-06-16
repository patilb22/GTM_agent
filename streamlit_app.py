from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


st.set_page_config(page_title="GTM Agent Dashboard", layout="wide")


@st.cache_data
def load_products() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "products.csv")


@st.cache_data
def load_text_file(name: str) -> str:
    return (DATA_DIR / name).read_text(encoding="utf-8")


def render_markdown_dataset(title: str, file_name: str) -> None:
    with st.expander(title, expanded=False):
        st.markdown(load_text_file(file_name))


def run_gtm_agent(input_brief: str) -> dict:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set for this Streamlit process.")

    sys.modules.pop("gtm_agent", None)
    from gtm_agent import app

    config = {"configurable": {"thread_id": "streamlit-gtm-dashboard"}}
    output = app.invoke({"input_brief": input_brief}, config=config)
    return normalize_agent_output(output)


def normalize_agent_output(output: dict) -> dict:
    interrupts = output.get("__interrupt__")
    if not interrupts:
        return output

    interrupt = interrupts[0]
    payload = getattr(interrupt, "value", interrupt)
    if not isinstance(payload, dict):
        return output

    return {
        **output,
        "suite": payload.get("suite", output.get("suite", {})),
        "review": payload.get("review_agent_feedback", output.get("review", {})),
        "human_review_task": payload.get("task", ""),
    }


st.title("GTM Agent Dashboard")

with st.sidebar:
    st.header("Settings")
    env_api_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        value=env_api_key,
        help="Used only for this local Streamlit session.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("OpenAI API key is available.")
    else:
        st.warning("OPENAI_API_KEY is not set.")

products = load_products()

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Input Dataset")
    st.dataframe(products, use_container_width=True, hide_index=True)
    render_markdown_dataset("Upcoming GTM Events", "events.md")
    render_markdown_dataset("Past Campaign Messaging Library", "past_campaigns.md")

with right:
    st.subheader("Run Agent")
    selected_product = st.selectbox("Product", products["name"].tolist())
    default_brief = f"Build a content suite for the {selected_product} Product Hunt launch."
    input_brief = st.text_area("Input brief", value=default_brief, height=120)

    run_clicked = st.button("Generate GTM Suite", type="primary")

    if run_clicked:
        with st.spinner("Generating content suite and review..."):
            try:
                st.session_state["agent_output"] = run_gtm_agent(input_brief)
                st.session_state["agent_error"] = None
            except Exception as exc:
                st.session_state["agent_output"] = None
                st.session_state["agent_error"] = str(exc)

    error = st.session_state.get("agent_error")
    output = st.session_state.get("agent_output")

    if error:
        st.error(error)
        st.info("Check that dependencies are installed and OPENAI_API_KEY is set before running the agent.")

    if output:
        st.subheader("Output Data")

        summary = {
            "Audience": output.get("audience", ""),
            "Tone": output.get("tone", ""),
            "Review score": output.get("review", {}).get("score", ""),
            "Approved": output.get("review", {}).get("approved", ""),
        }
        st.dataframe(pd.DataFrame([summary]), use_container_width=True, hide_index=True)

        suite = output.get("suite", {})
        tabs = st.tabs(["LinkedIn", "Email", "Blog", "Ads", "Review", "Context"])

        with tabs[0]:
            st.write(suite.get("linkedin_post", ""))
        with tabs[1]:
            st.write(suite.get("promo_email", ""))
        with tabs[2]:
            st.write(suite.get("blog_draft", ""))
        with tabs[3]:
            st.dataframe(
                pd.DataFrame({"ad_copy_variation": suite.get("ad_copy_variations", [])}),
                use_container_width=True,
                hide_index=True,
            )
        with tabs[4]:
            review = output.get("review", {})
            st.write(review.get("revision_notes", ""))
            st.dataframe(
                pd.DataFrame({"issue": review.get("issues", [])}),
                use_container_width=True,
                hide_index=True,
            )
        with tabs[5]:
            st.text(output.get("retrieved_context", ""))
