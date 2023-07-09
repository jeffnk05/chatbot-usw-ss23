import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import os

from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV,  # next to api key in console
)


template = """

You are an AI tasked with evaluating stories similar to those on the subreddit r/amitheasshole. Please analyze the following story and provide your unbiased judgment and advice:

{history}

Human: {human_input}

Based on the story provided, please rate the behavior of the individual(s) involved using the following terms: YTA (You're the Asshole), YWBTA (You Would Be the Asshole), NTA (Not the Asshole), YWNBTA (You Would Not be the Asshole), ESH (Everyone Sucks Here), NAH (No Assholes Here), INFO (Not Enough Info).

Please provide a well-reasoned and balanced response, considering various perspectives and ethical considerations. Take into account the intentions, actions, and consequences of the individuals involved.

If you require additional information to provide a more accurate judgment, please ask for it specifically by using the term INFO. The user will then provide the requested information.

Feel free to ask me any questions about the judgment or request further clarification based on the provided story or the information exchanged in our conversation history.

Finally, offer constructive advice on how the individual(s) can improve the situation, learn from their actions, or make amends, if applicable.

Please provide your response, speaking directly to the user (You).

In case the user asks follow-up questions, DO NOT include the judgment, just answer the question.

"""

treshhold = 0.9

st.title('Am I The Asshole')

st.sidebar.markdown(
    "## How to use\n"
    "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"  
    "2. Describe your situation✍️\n"
    "3. Receive a verdict🧑‍⚖️\n"
)

openai_api_key = st.sidebar.text_input('OpenAI API Key')


def generate_response(input_text):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    index_name = PINECONE_INDEX
    vectorstore = Pinecone.from_existing_index(index_name, embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type="stuff",
                                     retriever=vectorstore.as_retriever())
    score = vectorstore.similarity_search_with_score(input_text)
    print(score[0][1])
    if score[0][1] >= treshhold:
        response = qa.run(input_text)
        st.info(response)
        print(response)
        print("vector version")

    else:
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        # Define the prompt template for the chat conversation
        prompt = PromptTemplate(
            input_variables=["history", "human_input"],
            template=template
        )

        # Create an instance of LLMChain for chat conversations
        chatgpt_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=2),
        )
        st.info(chatgpt_chain.predict(human_input=input_text))
        print("Prompt version")

with st.form('my_form'):
    text = st.text_area('Enter text:', 'Am i the asshole for...?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)
