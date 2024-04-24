import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from IPython.display import Markdown as md
import streamlit as st
from langchain_core.runnables import RunnablePassthrough


# Load the API Key
f = open(".api_key.txt")
GOOGLE_API_KEY = f.read()

st.set_page_config(page_title="InsightForge🧑‍💻", page_icon="🤖", initial_sidebar_state="collapsed")

chat_model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-1.5-pro-latest")

st.markdown(
    """
    <style>
    body {
        border: 1px solid #03A9F4;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <span style='color: #ADD8E6;'>
        <h1>InsightForge: Discover Answers in Context with AI💬🧑‍💻</h1>
    </span>


    """,
    unsafe_allow_html=True
)

# Streamlit UI

st.subheader("Get an answer based Contextual Question Answering with LangChain RAG System") 



embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Define chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are an AI Bot designed to provide helpful answers based on the context provided by the user."""),
    HumanMessagePromptTemplate.from_template("""Please answer the question based on the given context.
    
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

output_parser = StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

# Input field for user question
user_question = st.text_input("Enter your question here:")

# Button to trigger question answering
if st.button("GENERATE👆"):
    if user_question:
        response = rag_chain.invoke(user_question)
        st.markdown(response)
    else:
        st.warning("Please enter a question.")