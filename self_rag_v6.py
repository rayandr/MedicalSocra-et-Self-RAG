"""
Self-RAG (Retrieval-Augmented Generation) System with Self-Reflection

This implementation creates an advanced RAG system that continuously evaluates and improves its own outputs.
Instead of just retrieving documents and generating answers, it uses multiple LLM agents to:
1. Route questions to appropriate data sources
2. Evaluate document relevance
3. Check for hallucinations in generated answers
4. Verify answer quality and completeness

The system uses LangGraph to create a workflow that can adapt and improve through self-assessment.
"""

# ============================================================================
# IMPORTS AND ENVIRONMENT SETUP
# ============================================================================

import os
from dotenv import load_dotenv

# Core LangChain imports for LLM interactions and document processing
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# Document processing and vector store imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Output parsing and web search imports
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from typing import List

# Graph workflow imports
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from typing import Literal, List, Union

# Load environment variables from .env file
load_dotenv()

# Get API keys for various services
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # For web search functionality
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")   # For additional search capabilities
GROQ_API_KEY = os.getenv("GROQ_API_KEY")       # For LLM access via Groq

# Environment configuration to prevent warnings and set user agent
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevents HuggingFace tokenizer warnings
os.environ["USER_AGENT"] = "my-langchain-app/0.1"  # Sets custom user agent for web requests

# ============================================================================
# QUESTION ROUTING SYSTEM
# ============================================================================

# Define the data model for structured output from the routing LLM
class RouteQuery(BaseModel):
    """
    Data model for routing decisions.
    Forces the LLM to return either "vectorstore" or "websearch" as a structured response.
    """
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

# Initialize the main LLM (Llama 3.3 70B via Groq) with structured output capability
llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")  # Temperature 0 for consistent outputs
structured_llm_router = llm.with_structured_output(RouteQuery)  # Wraps LLM to enforce structured output

# Define the system prompt for question routing
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to diabetes.
Use the vectorstore for questions on these topics. For all else, use web-search."""

# Create the routing prompt template
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),  # System instructions
    ("human", "{question}"),  # User question placeholder
])

# Create the routing chain by combining prompt and structured LLM
question_router = route_prompt | structured_llm_router

# Example usage (commented out):
# print(question_router.invoke({"question": "Who will the Bears draft first in the NFL draft?"}))
# print(question_router.invoke({"question": "What are the types of agent memory?"}))
# print(question_router.invoke({"question": "What is treatment for type 1 diabetes?"}))

# ============================================================================
# DOCUMENT LOADING AND PROCESSING
# ============================================================================

# Define document sources (mixed URLs and local files)
sources: List[Union[str]] = [
    "https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451",  # Web source
    "https://www.who.int/news-room/fact-sheets/detail/diabetes",  # Web source
    "/Users/benoitchambers/Documents/langgraph/chatbot_data/diabetes_questions_answers.csv"  # Local CSV file
]

# Load documents dynamically based on file type/URL
docs = []

for src in sources:
    if src.startswith("http"):
        # Handle web URLs
        loader = WebBaseLoader(src)
        docs.extend(loader.load())
    elif src.endswith(".pdf") and os.path.exists(src):
        # Handle PDF files
        loader = PyPDFLoader(src)
        docs.extend(loader.load())
    elif src.endswith(".csv") and os.path.exists(src):
        # Handle CSV files
        loader = CSVLoader(file_path=src)
        docs.extend(loader.load())
    else:
        # Skip unsupported or missing sources
        print(f"Skipping unsupported or missing source: {src}")



# ============================================================================
# VECTOR STORE CREATION
# ============================================================================

# Split documents into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,  # Each chunk is 250 characters
    chunk_overlap=0  # No overlap between chunks
)
doc_splits = text_splitter.split_documents(docs)

# Create embeddings using HuggingFace model
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")  # High-quality multilingual embeddings

# Create vector store with the processed documents
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="adv-rag-chroma",  # Collection name for organization
    embedding=embeddings,
)

# Create retriever interface for similarity search
retriever = vectorstore.as_retriever()

# ============================================================================
# EVALUATION COMPONENTS
# ============================================================================

# ============================================================================
# 1. DOCUMENT RELEVANCE GRADER
# ============================================================================

# Data model for document relevance evaluation
class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    Forces LLM to return "yes" or "no" for document relevance.
    """
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# Create structured LLM for document grading
structured_llm_grader_docs = llm.with_structured_output(GradeDocuments)

# System prompt for document relevance evaluation
system = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

# Create the grading prompt template
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])

# Create the document relevance grading chain
retrieval_grader_relevance = grade_prompt | structured_llm_grader_docs


# ============================================================================
# 2. RAG GENERATION CHAIN
# ============================================================================

# Prompt template for RAG answer generation
prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""
)
    
# Create RAG chain by combining prompt, LLM, and output parser
rag_chain = prompt | llm | StrOutputParser()

# Test RAG generation:
# generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)

# ============================================================================
# 3. HALLUCINATION GRADER
# ============================================================================

# Data model for hallucination detection
class GradeHallucinations(BaseModel):
    """
    Binary score for hallucination present in generation answer.
    Checks if the generated answer is supported by the provided documents.
    """
    binary_score: str = Field(description="Don't consider calling external APIs for additional information. Answer is supported by the facts, 'yes' or 'no'.")
 
# Create structured LLM for hallucination grading
structured_llm_grader_hallucination = llm.with_structured_output(GradeHallucinations)
 
# System prompt for hallucination detection
system = """You are a grader assessing whether an LLM generation is supported by a set of retrieved facts. 
Restrict yourself to give a binary score, either 'yes' or 'no'. If the answer is supported or partially supported by the set of facts, consider it a yes. 
Don't consider calling external APIs for additional information as consistent with the facts."""

# Create hallucination grading prompt
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
])

# Create hallucination grading chain
hallucination_grader = hallucination_prompt | structured_llm_grader_hallucination

# Test hallucination grader (commented out):
# hallucination_grader.invoke({"documents": docs, "generation": generation})

# ============================================================================
# 4. ANSWER QUALITY GRADER
# ============================================================================

# Data model for answer quality evaluation
class GradeAnswer(BaseModel):
    """
    Binary score to assess answer addresses question.
    Verifies that the generated answer actually addresses the user's question.
    """
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

# Create structured LLM for answer quality grading
structured_llm_grader_answer = llm.with_structured_output(GradeAnswer)

# System prompt for answer quality evaluation
system = """You are a grader assessing whether an answer addresses / resolves a question 
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

# Create answer quality grading prompt
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
])

# Create answer quality grading chain
answer_grader = answer_prompt | structured_llm_grader_answer

# Test answer grader:
# answer_grader.invoke({"question": question,"generation": generation})

# ============================================================================
# WEB SEARCH TOOL
# ============================================================================

# Initialize web search tool for getting additional information
web_search_tool = TavilySearchResults(k=3)  # Returns top 3 search results

# ============================================================================
# GRAPH STATE DEFINITION
# ============================================================================

class GraphState(TypedDict):
    """
    Represents the state of our graph that flows through all nodes.
    
    This is the data structure that gets passed between different nodes
    in the workflow, maintaining the context and results throughout
    the entire process.
    
    Attributes:
        question: The original user question (immutable)
        generation: The LLM's generated answer (updated through workflow)
        web_search: Flag indicating if web search is needed ("Yes"/"No")
        documents: List of retrieved documents (updated through workflow)
    """
    question : str
    generation : str
    web_search : str
    documents : List[str]




# ============================================================================
# GRAPH NODES (WORKFLOW FUNCTIONS)
# ============================================================================

def retrieve(state):
    """
    Retrieve documents from vector store based on the user's question.
    
    This node performs similarity search in the vector database to find
    the most relevant documents for answering the user's question.
    
    Args:
        state (dict): The current graph state containing the question
        
    Returns:
        state (dict): Updated state with retrieved documents
    """
    print("---RETRIEVE from Vector Store DB---")
    question = state["question"]

    # Perform similarity search using the question
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer using RAG on retrieved documents.
    
    This node takes the retrieved documents and the user's question,
    then uses the RAG chain to generate a comprehensive answer.
    
    Args:
        state (dict): The current graph state containing question and documents
        
    Returns:
        state (dict): Updated state with generated answer
    """
    print("---GENERATE Answer---")
    question = state["question"]
    documents = state["documents"]
    
    # Generate answer using RAG chain
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Evaluate whether retrieved documents are relevant to the question.
    
    This node uses an LLM to assess the relevance of each retrieved document.
    If any document is irrelevant, it sets a flag to trigger web search
    for additional information.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        state (dict): Filtered documents and web search flag
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Evaluate each document for relevance
    filtered_docs = []
    web_search = "No"
    for d in documents:
        # Use LLM to grade document relevance
        score = retrieval_grader_relevance.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        
        # Document is relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document is not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # Don't include irrelevant document, set flag for web search
            web_search = "Yes"
            continue
            
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state):
    """
    Perform web search to get additional information.
    
    This node uses the Tavily search tool to find relevant information
    from the web and adds it to the document collection.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        state (dict): Updated state with web search results
    """
    print("---WEB SEARCH. Append to vector store db---")
    question = state["question"]
    documents = state["documents"]

    # Perform web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])  # Combine search results
    web_results = Document(page_content=web_results)  # Convert to Document object
    
    # Add web results to existing documents
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
        
    return {"documents": documents, "question": question}




# ============================================================================
# DECISION FUNCTIONS (GRAPH EDGES)
# ============================================================================

def route_question(state):
    """
    Route the question to either web search or vector store retrieval.
    
    This function uses an LLM to intelligently decide whether the question
    should be answered using the existing vector store (for domain-specific
    questions) or web search (for general questions).
    
    Args:
        state (dict): The current graph state
        
    Returns:
        str: Next node to call ("websearch" or "vectorstore")
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    
    # Use LLM to determine appropriate data source
    source = question_router.invoke({"question": question})   
    
    if source.datasource == 'websearch':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source.datasource == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Decide whether to generate an answer or get more information via web search.
    
    This function evaluates the results of document grading and decides
    whether there are enough relevant documents to generate an answer,
    or if more information is needed from the web.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        str: Next node to call ("websearch" or "generate")
    """
    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # Some documents were irrelevant, get more information
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # Have relevant documents, proceed to generate answer
        print("---DECISION: GENERATE---")
        return "generate"



def grade_generation_v_documents_and_question(state):
    """
    Two-stage evaluation of the generated answer.
    
    This function performs a comprehensive evaluation of the generated answer:
    1. First checks if the answer is grounded in the provided documents (no hallucinations)
    2. Then checks if the answer actually addresses the user's question
    
    Args:
        state (dict): The current graph state
        
    Returns:
        str: Decision for next node ("useful", "not useful", or "not supported")
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Stage 1: Check for hallucinations
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    # If answer is grounded in documents
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        
        # Stage 2: Check if answer addresses the question
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score.binary_score
        
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"  # Success - answer is good
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"  # Answer doesn't address question
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"  # Hallucination detected, retry




# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

# Create the main workflow graph
workflow = StateGraph(GraphState)

# Define the nodes (functions) in the workflow
workflow.add_node("websearch", web_search)      # Web search functionality
workflow.add_node("retrieve", retrieve)         # Document retrieval
workflow.add_node("grade_documents", grade_documents)  # Document relevance grading
workflow.add_node("generate", generate)         # Answer generation

# Define direct edges (fixed paths)
workflow.add_edge("websearch", "generate")      # Web search → Generate answer
workflow.add_edge("retrieve", "grade_documents")  # Retrieve → Grade documents

# Set the conditional entry point (first decision)
workflow.set_conditional_entry_point(
    route_question,  # Function that decides initial path
    {
        "websearch": "websearch",    # Route to web search
        "vectorstore": "retrieve",   # Route to vector store retrieval
    },
)

# Add conditional edge after document grading
workflow.add_conditional_edges(
    "grade_documents",  # From this node
    decide_to_generate,  # Use this function to decide
    {
        "websearch": "websearch",  # If decision is "websearch"
        "generate": "generate",    # If decision is "generate"
    },
)

# Add conditional edge after answer generation
workflow.add_conditional_edges(
    "generate",  # From this node
    grade_generation_v_documents_and_question,  # Use this function to decide
    {
        "not supported": "generate",  # Retry generation if not grounded
        "useful": END,                # End if answer is good
        "not useful": "websearch",    # Get more info if answer doesn't address question
    },
)

# Compile the workflow into an executable application
app = workflow.compile()

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

# Example usage (commented out):
# from pprint import pprint
# 
# # Test with a general question (should go to web search)
# inputs = {"question": "Which player was selected as the first pick in the NBA 2023 draft?"}
# 
# # Test with a domain-specific question (should use vector store)
# inputs = {"question": "What is treatment for type 1 diabetes?"}
# 
# # Run the workflow
# for output in app.stream(inputs):
#     for key, value in output.items():
#         pprint(f"Finished running: {key}:")
#         if key == "generate":
#             pprint(f"Final answer: {value['generation']}")

