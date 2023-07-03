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

index_name = PINECONE_INDEX

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vectorstore = Pinecone.from_existing_index(index_name, embeddings)
query = "Am i the asshole for not inviting myy mother to my wedding?"
docs = vectorstore.similarity_search_with_score(query)
# print(docs[0][1])
# print(vectorstore.similarity_search(query))

qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="map_reduce",
                                 retriever=vectorstore.as_retriever())

# print(qa.run(query))

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

st.title('Am I The Asshole')

openai_api_key = st.sidebar.text_input('OpenAI API Key')

treshhold = 0.9


def generate_response(input_text):
    score = vectorstore.similarity_search_with_score(input_text)
    if score[0][1] >= treshhold:
        st.info(qa.run(input_text))

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


with st.form('my_form'):
    text = st.text_area('Enter text:', 'Am i the asshole for...?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)
