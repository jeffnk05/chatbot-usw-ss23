# Am I The Asshole

**Authors:** Melissa Tro (Student ID: 571267), Jeff Nkatiah Edjekoomhene (Student ID: 582338)

This is a Streamlit app that uses the LangChain library to analyze and provide judgments on stories similar to those on the subreddit r/amitheasshole. The app evaluates the behavior of individuals involved in a given story and provides unbiased judgments and advice.

## Installation

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.

## Usage

1. Set up your Pinecone environment by following the instructions provided in the Pinecone console.
2. Fill in the required secrets in `secrets.toml`.
3. Run the Streamlit app using `streamlit run main.py`.
4. Enter your OpenAI API key in the sidebar.
5. Describe the situation in the text area.
6. Click the "Submit" button to generate a verdict.

## Deployed App

The app is deployed and can be accessed at [https://chatbot-usw-ss23.streamlit.app/](https://chatbot-usw-ss23.streamlit.app/).

## Components

### LangChain

The LangChain library provides the following modules:

- `llms`: OpenAI language models integration.
- `prompts`: Prompt templates for chat conversations.
- `chains`: Language model chains for chat conversations.
- `memory`: Conversation memory management.
- `embeddings.openai`: OpenAI embeddings integration.
- `vectorstores`: Pinecone vector store integration.

### Pinecone

Pinecone is used for semantic similarity search in the application. It provides the following features:

- Initialization of Pinecone with the API key and environment.
- Creation of a vector store using Pinecone.
- Similarity search and retrieval of relevant information.
