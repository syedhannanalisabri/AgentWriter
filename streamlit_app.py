import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain.llms import HuggingFaceHub

# ✅ Force CrewAI to use LangChain, not LiteLLM
os.environ["CREWAI_LLM_PROVIDER"] = "langchain"

# ✅ Load environment variables
load_dotenv()

# ✅ Initialize HuggingFace LLM (free models)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # You can also try: tiiuae/falcon-7b-instruct
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
)

# ✅ Create agents
def create_agents():
    planner = Agent(
        role="Content Planner",
        goal="Plan engaging and factually accurate content on {topic}",
        backstory="You're a strategic thinker who outlines the blog based on trends and SEO.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )
    writer = Agent(
        role="Content Writer",
        goal="Write insightful and structured content based on planner's outline.",
        backstory="You're a creative writer who turns ideas into compelling blog posts.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )
    editor = Agent(
        role="Editor",
        goal="Refine the blog post to ensure clarity, flow, and consistency.",
        backstory="You're an expert editor making the content professional and publication-ready.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )
    return planner, writer, editor

# ✅ Create tasks
def create_tasks(planner, writer, editor):
    plan = Task(
        description="Create a detailed content outline for {topic} including intro, key points, audience, and SEO keywords.",
        expected_output="A structured outline for a blog post with headings and keywords.",
        agent=planner
    )
    write = Task(
        description="Write a blog post based on the planner's outline with markdown formatting.",
        expected_output="A blog post in markdown format with intro, body, and conclusion.",
        agent=writer
    )
    edit = Task(
        description="Edit the written blog post for grammar, clarity, and tone.",
        expected_output="A final polished markdown blog ready for publishing.",
        agent=editor
    )
    return [plan, write, edit]

# ✅ Streamlit UI
st.set_page_config(page_title="🧠 Free AI Blog Generator", layout="centered")
st.title("🧠 AI Blog Generator (Free with Hugging Face)")

topic = st.text_input("Enter a blog topic", placeholder="e.g. LLMs for DDoS Detection")

if st.button("Generate Blog"):
    if not topic.strip():
        st.warning("Please enter a topic")
        st.stop()

    with st.spinner("Agents working together..."):
        try:
            planner, writer, editor = create_agents()
            tasks = create_tasks(planner, writer, editor)
            crew = Crew(agents=[planner, writer, editor], tasks=tasks, verbose=True)
            result = crew.kickoff(inputs={"topic": topic})

            st.success("✅ Blog generated successfully!")
            st.markdown(result, unsafe_allow_html=True)
            st.download_button("💾 Download as Markdown", result, file_name="blog.md")
        except Exception as e:
            st.error(f"❌ Error: {e}")

