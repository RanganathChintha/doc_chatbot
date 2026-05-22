# rag/rag_chain.py

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from config import GROQ_API_KEY, LLM_MODEL

def get_llm() -> ChatGroq:
    """
    Initialize ChatGroq with the specified LLM model.
    """
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.2
    )
    print(f"✅ LLM loaded: {LLM_MODEL} via ChatGroq")
    return llm

def build_rag_chain(retriever, llm: ChatGroq):
    """
    Build the full RAG chain:
    Query → Retrieve relevant chunks → Format prompt → LLM → Answer
    """
    prompt_template = ChatPromptTemplate.from_template("""
You are a helpful and knowledgeable assistant.
Use ONLY the context provided below to answer the user's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain