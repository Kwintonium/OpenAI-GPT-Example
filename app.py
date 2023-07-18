# Import packages
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

# App framework (streamlit)
st.title('Youtube Video Idea Helper 💯👌')
prompt = st.text_input('Insert your prompt here')

# Create title prompt for GPT
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='Write me a youtube video title about {topic}'
)

# Create script prompt based on title response and wikipedia research
script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template='Write me a youtube video script based on this title: {title} while leveraging this wikipedia research:{wikipedia_research}'
)

# Memory (to view on webpage)
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# LLMs (OpenAI) configured to prompt templates
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper() # Init wiki wrapper

# Show text to screen if prompt
if prompt:
    # Run the title chain and wiki API based off of prompts
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    # Run the script chain based off of title response and wiki research
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    # Post resulting responses from GPT model
    st.write(title)
    st.write(script)

    # Store the memory and allow user to view it
    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)