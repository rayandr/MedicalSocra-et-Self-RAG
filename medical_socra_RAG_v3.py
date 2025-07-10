"""
Medical Socratic RAG (Retrieval-Augmented Generation) System

This implementation creates an advanced medical consultation system that simulates
a Socratic dialogue between two medical professionals with opposing perspectives.
The system uses RAG (Retrieval-Augmented Generation) to enhance medical knowledge
and CRIT evaluation to assess the quality of the discussion.

Key Features:
1. Dual Perspective Analysis: Two AI doctors (Primary and Alternative) with different viewpoints
2. Socratic Dialogue: Structured debate format with multiple rounds of discussion
3. RAG Enhancement: Retrieves relevant medical documents to support arguments
4. CRIT Evaluation: Continuous assessment of discussion quality
5. Contentiousness Control: Gradually reduces debate intensity over rounds
6. Medical Document Support: Handles TXT, CSV, and PDF medical documents

The system creates a comprehensive medical analysis by exploring both supporting
and questioning perspectives on potential diagnoses.
"""

# ============================================================================
# IMPORTS AND ENVIRONMENT SETUP
# ============================================================================

import openai
import re
import os
from dotenv import load_dotenv
import time
import json
from typing import List, TypedDict, Dict, Any, Optional

# LangGraph and RAG components for document processing and workflow management
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader

# Import the CRIT evaluation class for assessing discussion quality
from crit import Crit

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key for LLM interactions
OPENAI_API_KEY_STAGE = os.getenv("OPENAI_API_KEY_STAGE")

# ============================================================================
# GRAPH STATE DEFINITION
# ============================================================================

class MedicalGraphState(TypedDict):
    """
    Represents the state of our MedicalSocraRAG graph throughout the discussion.
    
    This data structure maintains all the information needed for the medical
    discussion, including symptoms, diagnoses, discussion history, and evaluation
    metrics. It flows through all nodes in the workflow.
    
    Attributes:
        symptoms: Patient symptoms (input from user)
        primary_diagnoses: List of primary/supporting diagnoses
        alternative_diagnoses: List of alternative/questioning diagnoses
        contentiousness: Current debate intensity level (starts high, decreases)
        documents: Retrieved medical documents for RAG support
        generation: Current generated content
        full_discussion: Complete discussion history with all exchanges
        current_round: Current round number in the discussion
        crit_score: Current CRIT evaluation score (0-10)
        previous_crit_score: Previous round's CRIT score for comparison
    """
    symptoms: str
    primary_diagnoses: List[str]
    alternative_diagnoses: List[str]
    contentiousness: float
    documents: List[Document]
    generation: str
    full_discussion: List[Dict[str, str]]
    current_round: int
    crit_score: float
    previous_crit_score: float

# ============================================================================
# MAIN MEDICAL SOCRATIC RAG CLASS
# ============================================================================

class MedicalSocraRAG:
    """
    Medical Socratic RAG system that creates a structured debate between
    two AI medical professionals with opposing perspectives.
    
    This class implements a sophisticated medical consultation system that:
    1. Analyzes patient symptoms from multiple perspectives
    2. Engages in structured medical debate
    3. Uses RAG to enhance arguments with medical knowledge
    4. Evaluates discussion quality using CRIT
    5. Provides comprehensive medical analysis
    """
    
    def __init__(self, crit_instance, medical_docs_path="./medical_docs"):
        """
        Initialize the Medical Socra RAG system.
        
        Args:
            crit_instance: An instance of the Crit class for evaluating discussions
            medical_docs_path: Path to medical documents for RAG enhancement
        """
        # Store the CRIT evaluator for discussion quality assessment
        self.crit = crit_instance
        
        # Initialize OpenAI client for LLM interactions
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY_STAGE)
        
        # Set path for medical documents
        self.medical_docs_path = medical_docs_path
        
        # Initialize workflow (will be built later)
        self.workflow = None
        
        # Set up the vector store and retriever for medical knowledge
        self.retriever = self.setup_retriever()
        
        # Build the LangGraph workflow
        self.build_workflow()
        
        print("Initialized Medical Socra RAG with Crit instance and document retrieval")

    def setup_retriever(self):
        """
        Set up the document retriever for medical knowledge with support for TXT, CSV, and PDF.
        
        This method creates a vector store from medical documents to enable
        semantic search for relevant medical information during discussions.
        
        Returns:
            Retriever object or None if setup fails
        """
        # Check if medical docs directory exists
        if not os.path.exists(self.medical_docs_path):
            print(f"Medical docs directory not found: {self.medical_docs_path}")
            print("Will proceed without document retrieval capability")
            return None
            
        try:
            # Load documents from different file formats
            docs = []
            
            # Process each file in the medical documents directory
            for filename in os.listdir(self.medical_docs_path):
                file_path = os.path.join(self.medical_docs_path, filename)
                
                # Process based on file extension
                if filename.lower().endswith('.txt'):
                    print(f"Loading text file: {filename}")
                    loader = TextLoader(file_path)
                    docs.extend(loader.load())
                    
                elif filename.lower().endswith('.csv'):
                    print(f"Loading CSV file: {filename}")
                    loader = CSVLoader(
                        file_path,
                        csv_args={
                            'delimiter': ',',
                            'quotechar': '"',
                            'fieldnames': None  # Use first row as headers
                        }
                    )
                    docs.extend(loader.load())
                    
                elif filename.lower().endswith('.pdf'):
                    print(f"Loading PDF file: {filename}")
                    loader = PyPDFLoader(file_path)
                    docs.extend(loader.load())
            
            # Check if any documents were loaded
            if not docs:
                print("No supported documents found in the directory")
                return None
                
            print(f"Loaded {len(docs)} line(s) from {self.medical_docs_path}")
            
            # Split documents into smaller chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=250,  # Each chunk is 250 characters
                chunk_overlap=50  # 50 character overlap between chunks
            )
            doc_splits = text_splitter.split_documents(docs)
            print(f"Split into {len(doc_splits)} chunks for processing")
            
            # Create embeddings using OpenAI for semantic search
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY_STAGE)
            
            # Create vector store with embeddings
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name="medical-rag-chroma",
                embedding=embeddings
            )
            
            # Configure retriever with search options
            return vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Retrieve top 5 most relevant documents
            )
        except Exception as e:
            print(f"Error setting up retriever: {e}")
            import traceback
            traceback.print_exc()
            return None

    def build_workflow(self):
        """
        Build the LangGraph workflow for MedicalSocraRAG.
        
        This method creates a directed graph that defines the flow of the medical
        discussion, including decision points for continuing or concluding the debate.
        """
        # Create the state graph with our defined state structure
        workflow = StateGraph(MedicalGraphState)
        
        # Define nodes (functions) for the workflow
        workflow.add_node("initialize_diagnoses", self.initialize_diagnoses)      # Generate initial diagnoses
        workflow.add_node("retrieve_medical_info", self.retrieve_medical_info)    # Get relevant medical documents
        workflow.add_node("generate_arguments", self.generate_arguments)          # Generate debate arguments
        workflow.add_node("evaluate_discussion", self.evaluate_discussion)        # Evaluate discussion quality
        workflow.add_node("generate_final_assessment", self.generate_final_assessment)  # Final conclusions
        
        # Define direct edges (fixed paths)
        workflow.add_edge("initialize_diagnoses", "retrieve_medical_info")        # Start → Get medical info
        workflow.add_edge("retrieve_medical_info", "generate_arguments")          # Get info → Generate arguments
        
        # Add conditional edge from generate_arguments (decision point)
        workflow.add_conditional_edges(
            "generate_arguments",
            self.should_continue_discussion,  # Decision function
            {
                "continue": "generate_arguments",      # Continue debate
                "evaluate": "evaluate_discussion",     # Evaluate current state
                "conclude": "generate_final_assessment"  # End discussion
            },
        )
        
        # Additional edges
        workflow.add_edge("evaluate_discussion", "generate_arguments")  # Evaluate → Continue
        workflow.add_edge("generate_final_assessment", END)            # Final assessment → End
        
        # Set the entry point of the workflow
        workflow.set_entry_point("initialize_diagnoses")
        
        # Compile the workflow into an executable application
        self.workflow = workflow.compile()

    def initialize_diagnoses(self, state: MedicalGraphState) -> MedicalGraphState:
        """
        Initialize primary and alternative diagnoses from symptoms.
        
        This is the first node in the workflow. It takes patient symptoms and
        generates two sets of potential diagnoses:
        1. Primary diagnoses (supporting perspective)
        2. Alternative diagnoses (questioning perspective)
        
        Args:
            state: Current graph state containing symptoms
            
        Returns:
            Updated state with initial diagnoses and discussion setup
        """
        symptoms = state["symptoms"]
        print(f"Generating medical interpretations for symptoms: {symptoms}")
        
        # Generate primary diagnoses (supporting perspective)
        primary_diagnoses = self.generate_subtopics(symptoms, "positive")
        
        # Generate alternative diagnoses (questioning perspective)
        alternative_diagnoses = self.generate_subtopics(symptoms, "negative")
        
        # Initialize discussion history
        full_discussion = []
        
        # Record the initial symptoms in the discussion
        moderator_entry = {
            "role": "Moderator",
            "content": f"Patient Symptoms for Analysis:\n{symptoms}"
        }
        full_discussion.append(moderator_entry)
        
        # Record the initial assessments from both perspectives
        primary_entry = {
            "role": "Dr. Primary",
            "content": f"Initial Assessment:\nBased on the symptoms, here are my primary diagnostic considerations:\n" + 
                       self.format_list_for_discussion(primary_diagnoses)
        }
        full_discussion.append(primary_entry)
        
        alternative_entry = {
            "role": "Dr. Alternative",
            "content": f"Initial Assessment:\nI'd like to suggest some alternative diagnoses that should also be considered:\n" + 
                       self.format_list_for_discussion(alternative_diagnoses)
        }
        full_discussion.append(alternative_entry)
        
        # Return updated state with initial setup
        return {
            "symptoms": symptoms,
            "primary_diagnoses": primary_diagnoses,
            "alternative_diagnoses": alternative_diagnoses,
            "contentiousness": 90,  # Start at 90% contentiousness (high debate intensity)
            "documents": [],
            "generation": "",
            "full_discussion": full_discussion,
            "current_round": 0,
            "crit_score": 0,
            "previous_crit_score": 0
        }

    def retrieve_medical_info(self, state: MedicalGraphState) -> MedicalGraphState:
        """
        Retrieve relevant medical information for the diagnoses.
        
        This node uses RAG to find relevant medical documents that can support
        or challenge the proposed diagnoses. It enhances the discussion with
        evidence-based medical knowledge.
        
        Args:
            state: Current graph state with diagnoses
            
        Returns:
            Updated state with retrieved medical documents
        """
        print("Retrieving relevant medical information")
        
        # Check if retriever is available
        if not self.retriever:
            print("No retriever available, skipping information retrieval")
            return state
        
        # Combine primary and alternative diagnoses for search
        all_diagnoses = state["primary_diagnoses"] + state["alternative_diagnoses"]
        
        # Create a query from the diagnoses and symptoms
        query = f"Patient symptoms: {state['symptoms']}. Potential diagnoses: {', '.join(all_diagnoses)}"
        
        # Retrieve relevant documents
        try:
            documents = self.retriever.invoke(query)
            print(f"Retrieved {len(documents)} relevant medical documents")
            
            # Update state with retrieved documents
            return {**state, "documents": documents}
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return state

    def generate_arguments(self, state: MedicalGraphState) -> MedicalGraphState:
        """
        Generate arguments for the current round of discussion.
        
        This is the core node that generates the medical debate. It creates
        arguments from both perspectives, gradually reducing contentiousness
        over rounds, and evaluates the discussion quality using CRIT.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with new arguments and evaluation scores
        """
        contentiousness = state["contentiousness"]
        current_round = state["current_round"]
        full_discussion = state["full_discussion"]
        
        print(f"Generating arguments for round {current_round} with contentiousness level: {contentiousness}%")
        
        # For round 0, generate initial arguments
        if current_round == 0:
            # Generate arguments for primary diagnoses (supporting perspective)
            primary_arguments = self.generate_medical_arguments(
                state["primary_diagnoses"], 
                contentiousness, 
                "positive",
                state["documents"]
            )
            
            # Generate arguments for alternative diagnoses (questioning perspective)
            alternative_arguments = self.generate_medical_arguments(
                state["alternative_diagnoses"], 
                contentiousness, 
                "negative",
                state["documents"]
            )
            
            # Record the initial arguments in the discussion
            primary_args_entry = {
                "role": "Dr. Primary",
                "content": f"Clinical Reasoning (contentiousness: {contentiousness}%):\n" + 
                           self.format_list_for_discussion(primary_arguments)
            }
            full_discussion.append(primary_args_entry)
            
            alternative_args_entry = {
                "role": "Dr. Alternative",
                "content": f"Clinical Reasoning (contentiousness: {contentiousness}%):\n" + 
                           self.format_list_for_discussion(alternative_arguments)
            }
            full_discussion.append(alternative_args_entry)
            
            # Create evaluation document for CRIT assessment
            document = self.create_evaluation_document(
                state["symptoms"], primary_arguments, alternative_arguments
            )
            
            # Evaluate with CRIT
            crit_score = self.crit.crit(document)
            
            # Record CRIT score
            moderator_score_entry = {
                "role": "Moderator",
                "content": f"CRIT Score: {crit_score}/10"
            }
            full_discussion.append(moderator_score_entry)
            
            # Update state
            return {
                **state,
                "full_discussion": full_discussion,
                "current_round": current_round + 1,
                "crit_score": crit_score,
                "previous_crit_score": 0,
                "generation": document
            }
        
        # For subsequent rounds, generate responses to previous arguments
        else:
            # Reduce contentiousness over time (debate becomes less intense)
            new_contentiousness = round(contentiousness / 1.2)  # Using delta=1.2
            
            # Add a moderator message for this round
            moderator_round_entry = {
                "role": "Moderator",
                "content": f"Discussion Round {current_round} (contentiousness Level: {new_contentiousness}%)"
            }
            full_discussion.append(moderator_round_entry)
            
            # Extract the last two arguments from the alternative perspective
            alternative_entries = [
                entry["content"] for entry in full_discussion 
                if entry["role"] == "Dr. Alternative"
            ]
            recent_alternative_points = self.extract_arguments_from_entry(
                alternative_entries[-1] if alternative_entries else ""
            )
            
            # Generate primary doctor's response to alternative points
            primary_response = self.generate_medical_arguments(
                state["primary_diagnoses"],
                new_contentiousness,
                "positive",
                state["documents"],
                round_num=current_round,
                previous_points=recent_alternative_points[-2:] if len(recent_alternative_points) >= 2 else recent_alternative_points
            )
            
            # Add Dr. Primary's response
            primary_response_entry = {
                "role": "Dr. Primary",
                "content": f"Response to Alternative Considerations (contentiousness: {new_contentiousness}%):\n" + 
                           self.format_list_for_discussion(primary_response)
            }
            full_discussion.append(primary_response_entry)
            
            # Extract the most recent arguments from the primary perspective
            primary_entries = [
                entry["content"] for entry in full_discussion 
                if entry["role"] == "Dr. Primary"
            ]
            recent_primary_points = self.extract_arguments_from_entry(
                primary_entries[-1] if primary_entries else ""
            )
            
            # Generate alternative doctor's response to primary points
            alternative_response = self.generate_medical_arguments(
                state["alternative_diagnoses"],
                new_contentiousness,
                "negative",
                state["documents"],
                round_num=current_round,
                previous_points=recent_primary_points
            )
            
            # Add Dr. Alternative's response
            alternative_response_entry = {
                "role": "Dr. Alternative",
                "content": f"Response to Primary Considerations (contentiousness: {new_contentiousness}%):\n" + 
                           self.format_list_for_discussion(alternative_response)
            }
            full_discussion.append(alternative_response_entry)
            
            # Create evaluation document with all arguments
            document = self.create_evaluation_document(
                state["symptoms"], 
                primary_response, 
                alternative_response
            )
            
            # Evaluate with CRIT
            previous_crit_score = state["crit_score"]
            current_crit_score = self.crit.crit(document)
            
            # Record CRIT score
            crit_score_entry = {
                "role": "Moderator",
                "content": f"CRIT Score: {current_crit_score}/10"
            }
            full_discussion.append(crit_score_entry)
            
            # Update state
            return {
                **state,
                "contentiousness": new_contentiousness,
                "full_discussion": full_discussion,
                "current_round": current_round + 1,
                "previous_crit_score": previous_crit_score,
                "crit_score": current_crit_score,
                "generation": document
            }


    def should_continue_discussion(self, state: MedicalGraphState) -> str:
        """
        Decide whether to continue the discussion or conclude.
        
        This decision function determines when the medical debate should end.
        It considers multiple factors including CRIT score trends, contentiousness
        levels, and maximum round limits.
        
        Args:
            state: Current graph state
            
        Returns:
            str: Decision ("continue", "evaluate", or "conclude")
        """
        contentiousness = state["contentiousness"]
        crit_score = state["crit_score"]
        previous_crit_score = state["previous_crit_score"]
        current_round = state["current_round"]
        
        # Check if CRIT score is decreasing (discussion quality is declining)
        if crit_score < previous_crit_score and current_round > 1:
            print("CRIT score decreased, moving to final assessment")
            
            # Add a note to discussion
            state["full_discussion"].append({
                "role": "Moderator",
                "content": "CRIT Score has decreased. Moving to final assessment."
            })
            
            return "conclude"
        
        # Check if contentiousness is too low (debate has become too mild)
        if contentiousness <= 10:
            print("Contentiousness too low, moving to final assessment")
            
            # Add a note to discussion
            state["full_discussion"].append({
                "role": "Moderator",
                "content": "Contentiousness has reached the minimum threshold. Moving to final assessment."
            })
            
            return "conclude"
        
        # Check if we've reached maximum rounds
        if current_round >= 3:
            print("Maximum rounds reached, moving to final assessment")
            
            # Add a note to discussion
            state["full_discussion"].append({
                "role": "Moderator",
                "content": "Maximum discussion rounds reached. Moving to final assessment."
            })
            
            return "conclude"
        
        # Continue discussion for another round
        return "continue"



    def evaluate_discussion(self, state: MedicalGraphState) -> MedicalGraphState:
        """
        Evaluate the current state of the discussion.
        
        This node provides a place for additional evaluation beyond CRIT.
        Currently a pass-through node that could be enhanced with more
        sophisticated evaluation metrics in the future.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state (currently unchanged)
        """
        
        return state




    def generate_final_assessment(self, state: MedicalGraphState) -> MedicalGraphState:
        """
        Generate final assessments from both perspectives.
        
        This is the final node that creates comprehensive conclusions from
        both medical perspectives, summarizing the entire discussion.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with final assessments
        """
        full_discussion = state["full_discussion"]
        contentiousness = state["contentiousness"]
        
        # Add final assessment prompt
        moderator_final_entry = {
            "role": "Moderator",
            "content": "Please provide your final assessments."
        }
        full_discussion.append(moderator_final_entry)
        
        # Extract recent alternative points for final response
        alternative_entries = [
            entry["content"] for entry in full_discussion 
            if entry["role"] == "Dr. Alternative"
        ]
        recent_alternative_points = self.extract_arguments_from_entry(
            alternative_entries[-1] if alternative_entries else ""
        )
        
        # Generate primary doctor's final assessment
        final_primary_arguments = self.generate_medical_arguments(
            state["primary_diagnoses"],
            contentiousness,
            "positive",
            state["documents"],
            round_num=state["current_round"],
            previous_points=recent_alternative_points[-3:] if len(recent_alternative_points) >= 3 else recent_alternative_points,
            is_final=True
        )
        
        # Add Dr. Primary's final assessment
        primary_final_entry = {
            "role": "Dr. Primary",
            "content": f"Final Assessment:\n" + self.format_list_for_discussion(final_primary_arguments)
        }
        full_discussion.append(primary_final_entry)
        
        # Extract recent primary points for final response
        primary_entries = [
            entry["content"] for entry in full_discussion 
            if entry["role"] == "Dr. Primary"
        ]
        recent_primary_points = self.extract_arguments_from_entry(
            primary_entries[-1] if primary_entries else ""
        )
        
        # Generate alternative doctor's final assessment
        final_alternative_arguments = self.generate_medical_arguments(
            state["alternative_diagnoses"],
            contentiousness,
            "negative",
            state["documents"],
            round_num=state["current_round"],
            previous_points=recent_primary_points,
            is_final=True
        )
        
        # Add Dr. Alternative's final assessment
        alternative_final_entry = {
            "role": "Dr. Alternative",
            "content": f"Final Assessment:\n" + self.format_list_for_discussion(final_alternative_arguments)
        }
        full_discussion.append(alternative_final_entry)
        
        # Add closing statement
        moderator_closing_entry = {
            "role": "Moderator",
            "content": "The medical discussion is now complete. Both perspectives have been presented with their supporting evidence."
        }
        full_discussion.append(moderator_closing_entry)
        
        # Create final document with combined assessments
        final_document = self.create_evaluation_document(
            state["symptoms"], 
            final_primary_arguments, 
            final_alternative_arguments
        )
        
        # Update state
        return {
            **state,
            "full_discussion": full_discussion,
            "generation": final_document
        }




    # ============================================================================
    # HELPER METHODS FOR ARGUMENT GENERATION AND PROCESSING
    # ============================================================================

    def generate_subtopics(self, symptoms, perspective="positive"):
        """
        Generate potential diagnoses/interpretations from symptoms.
        
        This method uses an LLM to analyze symptoms and generate potential
        medical interpretations from either a supporting or questioning perspective.
        
        Args:
            symptoms: Patient symptoms to analyze
            perspective: "positive" for supporting diagnoses, "negative" for alternatives
            
        Returns:
            List of potential diagnoses/interpretations
        """
        print(f"Generating {perspective} medical interpretations for symptoms")
        
        # Determine the approach based on perspective
        prompt_approach = "most likely" if perspective == "positive" else "alternative"
        
        # Create the prompt for diagnosis generation
        prompt = f"""As a medical professional, analyze these symptoms and identify 3-5 {prompt_approach} potential diagnoses or medical interpretations:

        Symptoms: {symptoms}
        
        For each potential diagnosis:
        1. Provide the name of the possible condition
        2. Explain how these symptoms align with this condition
        3. Note any key diagnostic indicators that would strengthen this interpretation
        
        Format each as a concise bullet point."""
        
        # Generate response using LLM
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical consultation AI assistant helping analyze patient symptoms."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract bullet points from the response
        subtopics = self.extract_bullets(response.choices[0].message.content)
        print(f"Generated {len(subtopics)} potential medical interpretations")
        
        return subtopics




    def generate_medical_arguments(self, diagnoses, contentiousness, 
                                   perspective="positive", documents=None,
                                   round_num=0, previous_points=None, is_final=False):
        """
        Generate medical reasoning with augmented information from documents.
        
        This is the core method that generates medical arguments from either
        a supporting or questioning perspective. It can incorporate RAG-enhanced
        information and respond to previous arguments.
        
        Args:
            diagnoses: List of diagnoses to argue for/against
            contentiousness: Current debate intensity level
            perspective: "positive" for supporting, "negative" for questioning
            documents: Retrieved medical documents for RAG enhancement
            round_num: Current round number
            previous_points: Points from opponent to respond to
            is_final: Whether this is the final assessment
            
        Returns:
            List of medical reasoning points
        """
        print(f"Generating {perspective} medical reasoning with contentiousness level: {contentiousness}%")
        
        # Determine the reasoning approach based on perspective
        reasoning_approach = "supporting evidence for" if perspective == "positive" else "factors that may question"
        

        # Prepare context from documents if available
        document_context = ""
        if documents and len(documents) > 0:
            document_context = "Consider this relevant medical information:\n"
            for i, doc in enumerate(documents[:3], 1):  # Limit to first 3 documents
                document_context += f"Source {i}: {doc.page_content}\n\n"
        

        # Prepare reference points for responding to opponent
        reference_points = ""
        if previous_points and round_num > 0:
            reference_points = f"Directly address these specific points from your opponent:\n"
            for i, point in enumerate(previous_points, 1):
                # Extract just the first sentence for brevity
                if isinstance(point, str):
                    first_sentence = point.split('.')[0] + '.'
                    reference_points += f"{i}. {first_sentence} (Reference as 'Point {i}')\n"
        

        # Adjust prompt based on whether this is the final assessment
        purpose = "final assessment" if is_final else f"Round {round_num+1} arguments"
        focus_instruction = "Provide a comprehensive final assessment" if is_final else f"Focus on {2-round_num if round_num < 2 else 1} new arguments and address {round_num} opposing points"
        
        # Create the main prompt for argument generation
        prompt = f"""As a medical professional with {contentiousness}% contentiousness in your assessment, provide detailed {reasoning_approach} the following potential diagnoses based on the patient's symptoms.

        Potential diagnoses to analyze:
        {', '.join(diagnoses)}
        
        {document_context}
        
        {reference_points}
        
        For each point:
        1. Cite relevant medical knowledge or research when applicable
        2. Explain the clinical reasoning behind your analysis
        3. IF RESPONDING TO OPPONENT: Reference their points directly (e.g., "Regarding Point 2...")
        4. Suggest appropriate next steps for confirmation or treatment if relevant
        
        This is your {purpose}. {focus_instruction}."""
        
        # Include full history in the messages for better context
        messages = [
            {"role": "system", "content": "You are a medical professional in a discussion about patient diagnoses. Maintain consistent reasoning and directly reference opponent points when responding."}
        ]
        
        # Add the main prompt
        messages.append({"role": "user", "content": prompt})
        
        # Generate response using LLM
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        # Extract bullet points from the response
        arguments = self.extract_bullets(response.choices[0].message.content)
        print(f"Generated {len(arguments)} medical reasoning points")
        
        return arguments




    def extract_bullets(self, text):
        """
        Extract bullet points from text.
        
        This helper method parses LLM responses to extract individual
        bullet points or numbered items for structured output.
        
        Args:
            text: Text containing bullet points
            
        Returns:
            List of extracted bullet points
        """
        lines = text.strip().split("\n")
        bullets = []
        current_bullet = ""
        
        for line in lines:
            line = line.strip()
            # Check for bullet point indicators
            if line and (line.startswith('-') or line.startswith('*') or 
                        any(line.startswith(f"{i}.") for i in range(1, 10))):
                # Save previous bullet if any
                if current_bullet:
                    bullets.append(current_bullet)
                # Start new bullet
                current_bullet = line.lstrip('-*0123456789. \t')
            elif line and current_bullet:  # Continuation of previous bullet
                current_bullet += " " + line
                
        # Add the last bullet if exists
        if current_bullet:
            bullets.append(current_bullet)
                
        return bullets




    def extract_arguments_from_entry(self, entry_content):
        """
        Extract arguments from a discussion entry content.
        
        This method parses discussion entries to extract the actual
        arguments from the formatted content.
        
        Args:
            entry_content: Content of a discussion entry
            
        Returns:
            List of extracted arguments
        """
        # Skip the first line (title)
        content_lines = entry_content.strip().split('\n')[1:]
        content = '\n'.join(content_lines)
        
        # Extract bullet points
        return self.extract_bullets(content)




    def create_evaluation_document(self, symptoms, primary_analyses, alternative_analyses):
        """
        Create a document for CRIT evaluation with the medical analyses.
        
        This method formats the medical analyses into a document that
        can be evaluated by the CRIT system for quality assessment.
        
        Args:
            symptoms: Patient symptoms
            primary_analyses: Supporting medical analyses
            alternative_analyses: Questioning medical analyses
            
        Returns:
            Formatted evaluation document
        """
        doc = f"Patient Symptoms: {symptoms}\n\n"
        
        doc += "Primary Medical Interpretations:\n"
        for i, analysis in enumerate(primary_analyses, 1):
            doc += f"{i}. {analysis}\n\n"
        
        doc += "\nAlternative Medical Interpretations:\n"
        for i, analysis in enumerate(alternative_analyses, 1):
            doc += f"{i}. {analysis}\n\n"
            
        return doc




    def format_list_for_discussion(self, items):
        """
        Format a list of items for the discussion transcript.
        
        This method formats lists of arguments or diagnoses into
        a readable format for the discussion transcript.
        
        Args:
            items: List of items to format
            
        Returns:
            Formatted string
        """
        formatted = ""
        for i, item in enumerate(items, 1):
            formatted += f"{i}. {item}\n\n"
        return formatted




    def format_discussion_transcript(self, discussion):
        """
        Format the discussion for readable output.
        
        This method creates a formatted transcript of the entire
        medical discussion for easy reading and analysis.
        
        Args:
            discussion: List of discussion entries
            
        Returns:
            Formatted transcript string
        """
        transcript = ""
        for entry in discussion:
            transcript += f"\n\n{entry['role']}:\n"
            transcript += "=" * (len(entry['role']) + 1) + "\n"
            transcript += entry['content']
            transcript += "\n" + "-" * 80
        
        return transcript




    # ============================================================================
    # MAIN INTERFACE METHODS
    # ============================================================================

    def socra(self, symptoms):
        """
        Run the Medical Socra RAG process on the given symptoms.
        
        This is the main method that users call to start the medical
        consultation process. It runs the entire workflow from symptom
        analysis to final assessment.
        
        Args:
            symptoms: Patient symptoms to analyze
            
        Returns:
            Complete discussion transcript
        """
        print(f"Starting Medical Socra RAG process for symptoms: {symptoms}")
        
        # Prepare initial state
        initial_state = {
            "symptoms": symptoms,
            "primary_diagnoses": [],
            "alternative_diagnoses": [],
            "contentiousness": 90,
            "documents": [],
            "generation": "",
            "full_discussion": [],
            "current_round": 0,
            "crit_score": 0,
            "previous_crit_score": 0
        }
        
        # Run the workflow
        result = self.workflow.invoke(initial_state)
        
        # Extract and return the discussion
        return result["full_discussion"]
    



    def save_discussion(self, discussion, filename="medical_rag_discussion.json"):
        """
        Save the discussion to a file.
        
        This method allows users to save the complete medical discussion
        for later review or analysis.
        
        Args:
            discussion: Discussion to save
            filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(discussion, f, indent=2)
        print(f"Discussion saved to {filename}")




# ============================================================================
# MAIN EXECUTION AND INTERFACE FUNCTIONS
# ============================================================================

def main():
    """
    Main function for interactive medical consultation.
    
    This function provides an interactive interface for users to input
    symptoms and receive a comprehensive medical analysis.
    """
    
    # Initialize Crit first (for discussion quality evaluation)
    crit = Crit()
    
    # Initialize Medical Socra RAG with the Crit instance
    med_socra_rag = MedicalSocraRAG(crit)
    
    # Display welcome message
    print("\n" + "="*80)
    print("MEDICAL SYMPTOM ANALYZER WITH RAG")
    print("Enter the patient's symptoms below. Type 'done' on a new line when finished.")
    print("="*80)
    
    # Collect symptoms from user
    symptoms = []
    print("\nPlease enter symptoms (one per line, type 'done' when finished):")
    while True:
        symptom = input("> ").strip()
        if symptom.lower() == 'done':
            break
        if symptom:  # Only add non-empty lines
            symptoms.append(symptom)
    
    # Format symptoms for processing
    if not symptoms:
        print("No symptoms entered. Using example symptoms instead.")
        symptoms_text = """
        Persistent headache for 5 days, especially in the morning
        Slight fever (100.4°F)
        Fatigue
        Stiff neck
        Sensitivity to light
        Nausea but no vomiting
        """
    else:
        symptoms_text = "\n".join(symptoms)
        print("\nAnalyzing the following symptoms:")
        for i, symptom in enumerate(symptoms, 1):
            print(f"{i}. {symptom}")
    
    print("\nStarting medical analysis with RAG augmentation. This may take a few minutes...\n")
    
    # Run the analysis
    discussion = med_socra_rag.socra(symptoms_text)
    
    # Format and print the discussion
    transcript = med_socra_rag.format_discussion_transcript(discussion)
    print("\n\nMEDICAL CONSULTATION TRANSCRIPT (RAG-AUGMENTED):")
    print("=" * 80)
    print(transcript)
    print("=" * 80)



def medical_socra_run(symptoms_text: str) -> str:
    """
    Function called by app.py (for the interface).
    
    This function provides a simple interface for the medical consultation
    system, taking symptoms as input and returning a formatted conversation.
    
    Args:
        symptoms_text: Patient symptoms as text
        
    Returns:
        Formatted medical consultation transcript
    """
    crit = Crit()
    med_socra_rag = MedicalSocraRAG(crit)
    discussion = med_socra_rag.socra(symptoms_text)
    transcript = med_socra_rag.format_discussion_transcript(discussion)
    return transcript




# ============================================================================
# EXECUTION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

