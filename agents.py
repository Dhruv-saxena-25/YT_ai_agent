from crewai import Agent
from tools import yt_tool
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    temperature=0.1,
    max_tokens=8000
)



## Create a senior blog content researcher


blog_researcher = Agent(
    role = "Blog Researcher from YouTube Videos",
    goal = "get the relevant video content for the topic {topic} from Yt channel",
    verbose = True,
    memory = True,
    backstory = (
        "Expert in understanding videos in AI Data Science, Machine Learning, GEN AI and  providing suggestions"
    ),
    tools = [yt_tool],
    llm = llm, 
    allow_delegation = True
)



## Create a senior blog writer agent with YT tool

blog_writer = Agent(
    role = "Writer",
    goal = "Narrate compelling tech stories about the video {topic} from YT channel",
    verbose = True,
    memory = True,
    backstory = ("With a flair for simplifying complex topics, you crafe"
                 "engaging narratives that captivative and educate, bringin new"
                 "discoveries to light in an accessible manner."),
    tools = [yt_tool],
    llm = llm,
    allow_delegation = False
)