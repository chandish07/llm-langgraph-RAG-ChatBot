from typing import TypedDict, Sequence, List, Tuple
from langchain_core.documents import Document
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from pymongo import MongoClient
import numpy as np
import os
import sys
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import faiss
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
from bson import ObjectId

# Load environment variables
load_dotenv()

# Validate OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not found in environment variables")
    sys.exit(1)
if not api_key.startswith("sk-"):
    print("Error: Invalid OpenAI API key format. Key should start with 'sk-'")
    sys.exit(1)

try:
    # Initialize OpenAI embeddings with explicit API key
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-3-large"
    )
except Exception as e:
    print(f"Error initializing OpenAI embeddings: {str(e)}")
    sys.exit(1)

# Define state structure
class State(TypedDict):
    messages: Sequence[BaseMessage]
    context: str

def fetch_and_process_data():
    """Fetch data from MongoDB and process for vector store"""
    print("\nFetching data from MongoDB...")
    try:
        # Use a more robust connection approach
        client = MongoClient(
            "mongodb://localhost:27017/",
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000
        )
        
        # Test the connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB")
        
        db = client["TMdata"]
        collection = db["sample1"]
        
        # Create indexes for better performance
        try:
            collection.create_index([("total_obligation", -1)])
            collection.create_index([("base_and_all_options_value", -1)])
            collection.create_index([("potential_total_value_of_award", -1)])
            collection.create_index([("generated_unique_award_id", 1)])
        except Exception as e:
            print(f"Warning: Could not create indexes: {e}")
            # Continue even if indexes fail - they're not critical
        
        documents = []
        metadatas = []
        raw_texts = []
        
        cursor = collection.find().sort("total_obligation", -1)
        doc_count = 0
        for doc in cursor:
            doc_count += 1
            
            # Create detailed searchable text with structured information
            contract_info = [
                f"Contract ID: {doc.get('generated_unique_award_id', '')}",
                f"Contract Type: {doc.get('type', '')} - {doc.get('type_description', '')}",
                f"Contract Description: {doc.get('description', 'No description available.')}",
                f"Agency: {doc.get('awarding_agency_technomile_name', 'Unknown')}",
                f"NAICS Code: {doc.get('naics', 'Unknown')} - {doc.get('naics_description', 'Unknown')}",
                f"Total Obligation: ${doc.get('total_obligation', 0):,.2f}",
                
                # Vendor Information
                f"Vendor Name: {doc.get('recipient_name', '')}",
                f"Parent Company: {doc.get('parent_recipient_name', '')}",
                f"Vendor UEI: {doc.get('recipient_uei', '')}",
                f"Vendor Location: {doc.get('recipient_city', '')}, {doc.get('recipient_state', '')}, {doc.get('recipient_country', '')}",
                
                # Financial Information
                f"Contract Value: ${doc.get('total_obligation', 0):,.2f}",
                f"Base and Options Value: ${doc.get('base_and_all_options_value', 0):,.2f}",
                
                # Classification
                f"NAICS Code: {doc.get('naics', '')}",
                f"NAICS Description: {doc.get('naics_description', '')}",
                f"PSC Code: {doc.get('product_or_service_code', '')}",
                f"PSC Description: {doc.get('product_or_service_description', '')}",
                
                # Agency Information
                f"Agency: {doc.get('awarding_agency_name', '')}",
                f"Sub-Agency: {doc.get('awarding_sub_agency_name', '')}",
                f"Office: {doc.get('awarding_office_name', '')}",
                f"Major Command: {doc.get('awarding_major_command', '')}",
                
                # Location
                f"Place of Performance: {doc.get('pop_city_name', '')}, {doc.get('pop_state_name', '')}, {doc.get('pop_country_name', '')}",
                f"Contract Location: {doc.get('state_name', '')}, {doc.get('country_name', '')}",
                
                # Competition and Status
                f"Competition Type: {doc.get('extent_competed', '')}",
                f"Solicitation Procedure: {doc.get('solicitation_procedures', '')}",
                f"Contract Status: {doc.get('Status', '')}",
                
                # Dates
                f"Date Signed: {doc.get('date_signed', '')}",
                f"Period of Performance Start: {doc.get('period_of_performance_start_date', '')}",
                f"Period of Performance End: {doc.get('period_of_performance_end_date', '')}",
                
                # Additional Details
                f"Category Management: {doc.get('category_management', '')}",
                f"Set Aside Type: {doc.get('type_of_set_aside', '')}",
                f"Small Business Program: {doc.get('small_business_program', '')}"
            ]
            
            text = " | ".join(filter(None, contract_info))
            raw_texts.append(text)  # Store raw text for BM25
            
            # Create comprehensive metadata for better filtering and retrieval
            metadata = {
                "contract_id": doc.get("generated_unique_award_id", ""),
                "type": doc.get("type", ""),
                "type_description": doc.get("type_description", ""),
                "recipient_name": doc.get("recipient_name", ""),
                "parent_company": doc.get("parent_recipient_name", ""),
                "recipient_uei": doc.get("recipient_uei", ""),
                "total_obligation": float(doc.get("total_obligation", 0)),
                "base_value": float(doc.get("base_and_all_options_value", 0)),
                "potential_total_value": float(doc.get("potential_total_value_of_award", 0)),
                "naics": doc.get("naics", ""),
                "naics_description": doc.get("naics_description", ""),
                "psc_code": doc.get("product_or_service_code", ""),
                "psc_description": doc.get("product_or_service_description", ""),
                "agency": doc.get("awarding_agency_name", ""),
                "sub_agency": doc.get("awarding_sub_agency_name", ""),
                "office": doc.get("awarding_office_name", ""),
                "major_command": doc.get("awarding_major_command", ""),
                "pop_state": doc.get("pop_state_name", ""),
                "pop_country": doc.get("pop_country_name", ""),
                "competition": doc.get("extent_competed", ""),
                "status": doc.get("Status", ""),
                "date_signed": doc.get("date_signed", ""),
                "performance_start": doc.get("period_of_performance_start_date", ""),
                "performance_end": doc.get("period_of_performance_end_date", ""),
                "category": doc.get("category_management", ""),
                "set_aside": doc.get("type_of_set_aside", ""),
                "small_business": doc.get("small_business_program", ""),
                "description": doc.get("description", ""),
                "number_of_offers_received": doc.get("number_of_offers_received", 0)
            }
            
            print(f"Processing document {doc_count}: Contract {metadata['contract_id']}")
            documents.append(text)
            metadatas.append(metadata)
        
        print(f"\nSuccessfully processed {doc_count} documents")
        return documents, metadatas, raw_texts
        
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        print("Please ensure MongoDB is running and accessible")
        return [], [], []  # Return empty lists on error
    finally:
        if 'client' in locals():
            client.close()

def initialize_vector_store():
    """Initialize FAISS vector store with MongoDB data"""
    print("\nInitializing vector store...")
    documents, metadatas, raw_texts = fetch_and_process_data()
    
    if not documents:
        print("No documents found in MongoDB!")
        return None
    
    print(f"\nCreating FAISS index for {len(documents)} documents...")
    
    # Configure FAISS with IVF index for better search performance
    dimension = len(embeddings.embed_query("test"))  # Get embedding dimension
    
    # Create the index with cosine similarity (using L2 + inner product)
    vector_store = FAISS.from_texts(
        documents,
        embeddings,
        metadatas=metadatas,
        distance_strategy="COSINE"  # Use COSINE instead of INNER_PRODUCT
    )
    
    # Configure search parameters
    if len(documents) < 1000:
        nlist = 1  # Number of clusters for small datasets
    else:
        nlist = int(np.sqrt(len(documents)))  # Rule of thumb for number of clusters
    
    # Create IVF index with cosine similarity
    quantizer = faiss.IndexFlatIP(dimension)
    new_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Copy the data from the old index to the new one
    new_index.train(vector_store.index.reconstruct_n(0, len(documents)))
    vectors = []
    for i in range(len(documents)):
        vector = vector_store.index.reconstruct(i)
        vectors.append(vector)
    
    # Normalize all vectors at once for better performance
    vectors = np.array(vectors)
    faiss.normalize_L2(vectors)
    new_index.add(vectors)
    
    # Replace the old index with the new one
    vector_store.index = new_index
    
    print("Saving vector store locally...")
    vector_store.save_local("faiss_index")
    print("Vector store saved successfully!")
    return vector_store

def load_or_initialize_vector_store():
    """Load existing vector store or create new one"""
    if os.path.exists("faiss_index"):
        try:
            return FAISS.load_local(
                "faiss_index", 
                embeddings,
                allow_dangerous_deserialization=True  # Only enable if you trust the source
            )
        except Exception as e:
            print(f"Error loading existing vector store: {str(e)}")
            print("Creating new vector store...")
            return initialize_vector_store()
    return initialize_vector_store()

# Define HybridRetriever class first
class HybridRetriever:
    def __init__(self, vector_store, documents, alpha=0.5):
        """Initialize hybrid retriever with vector store and BM25."""
        self.vector_store = vector_store
        self.alpha = alpha
        
        # Optimize BM25 parameters
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(
            tokenized_docs,
            k1=1.5,  # Term frequency saturation parameter
            b=0.75,  # Length normalization parameter
            epsilon=0.25  # Smoothing parameter
        )
        self.documents = documents
    
    def hybrid_search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """Combine vector similarity and BM25 scores."""
        # Get vector similarity results with optimized parameters
        vector_results = self.vector_store.similarity_search_with_score(
            query,
            k=k,
            search_kwargs={
                "fetch_k": 100,  # Fetch more candidates
                "nprobe": 8,     # Search more clusters
                "ef_search": 64  # Increase search depth
            }
        )
        
        # Get BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores
        vector_scores = np.array([score for _, score in vector_results])
        norm_vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min())
        norm_bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        
        # Combine scores
        final_results = []
        for i, (doc, v_score) in enumerate(vector_results):
            combined_score = (
                self.alpha * (1 - norm_vector_scores[i]) +  # Convert similarity to distance
                (1 - self.alpha) * norm_bm25_scores[i]
            )
            final_results.append((doc, combined_score))
        
        # Sort by combined score
        return sorted(final_results, key=lambda x: x[1], reverse=True)[:k]

# Then initialize vector store and hybrid retriever
documents, metadatas, raw_texts = fetch_and_process_data()
vector_store = load_or_initialize_vector_store()
hybrid_retriever = HybridRetriever(vector_store, raw_texts, alpha=0.5)

def retrieve(state: State) -> State:
    """Retrieve relevant documents based on user query."""
    try:
        question = state["messages"][-1].content.strip()
        vector_store = load_or_initialize_vector_store()
        
        # Get more results for better coverage
        hybrid_retriever = HybridRetriever(vector_store, raw_texts)
        results = hybrid_retriever.hybrid_search(question, k=50)
        
        docs = [doc for doc, _ in results]
        
        context_parts = []
        for doc in docs:
            metadata = doc.metadata
            total_obligation = float(metadata.get('total_obligation', 0))
            base_value = float(metadata.get('base_and_all_options_value', 0))
            num_bidders = metadata.get('number_of_offers_received', 0)
            
            context_parts.append(
                f"""
                ═══════════════════════ CONTRACT SUMMARY ═══════════════════════
                
                📋 Basic Information
                ▸ Contract ID: {metadata.get('contract_id', 'N/A')}
                ▸ Agency: {metadata.get('agency', 'N/A')}
                ▸ Sub-Agency: {metadata.get('sub_agency', 'N/A')}
                
                💰 Financial Details
                ▸ Total Contract Value: ${total_obligation:,.2f}
                ▸ Base and Options Value: ${base_value:,.2f}
                ▸ Potential Total Value: ${float(metadata.get('potential_total_value', 0)):,.2f}
                
                🏢 Vendor Information
                ▸ Name: {metadata.get('recipient_name', 'N/A')}
                ▸ NAICS: {metadata.get('naics', 'N/A')}
                ▸ NAICS Description: {metadata.get('naics_description', 'N/A')}
                
                📊 Competition Details
                ▸ Number of Bidders: {num_bidders}
                ▸ Contract Status: {metadata.get('status', 'N/A')}
                
                📅 Important Dates
                ▸ Date Signed: {metadata.get('date_signed', 'N/A')}
                ▸ Start Date: {metadata.get('start_date', 'N/A')}
                ▸ End Date: {metadata.get('end_date', 'N/A')}
                ▸ Potential End Date: {metadata.get('potential_end_date', 'N/A')}
                
                📝 Description
                {metadata.get('description', 'N/A')}
                
                ════════════════════════════════════════════════════════════════
                """
            )
        
        context = "\n".join(context_parts)
        if not context.strip():
            context = "No relevant contracts found in the database."
        
        return {
            "messages": state["messages"],
            "context": context
        }
        
    except Exception as e:
        print(f"Error in retrieval: {str(e)}")
        return {
            "messages": state["messages"],
            "context": "Error retrieving contract information."
        }

def generate(state: State) -> State:
    """Generate response using chat model and context."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are GovSearchAI, a specialized assistant analyzing government contract data from our comprehensive database. Give direct, data-driven answers based solely on the information available in our system.
        Here is a **data dictionary** that explains the meaning of each field you might encounter:

        1. active_task_order (int):
        - Indicates if a task order is active or expired (0 = expired, otherwise active).

        2. agency_entity_level (int):
        - Describes the agency's corporate reporting level/hierarchy.

        3. award_id (string):
        - Unique ID for each individual award (grants, loans, etc.).

        4. awarding_agency_name (string):
        - The name of the top-level government department awarding the transaction.

        5. awarding_agency_office_name (string):
        - The name of the "level n" office within the awarding agency.

        6. awarding_agency_subtier_agency_code (string):
        - Identifier for the sub-tier agency awarding the transaction.

        7. awarding_agency_toptier_agency_code (string):
        - Identifier for the top-tier government department (used in TAFS).

        8. awarding_office_code (string):
        - Code for the office responsible for awarding the contract.

        9. awarding_sub_agency_name (string):
        - Name of the level 2 organization awarding or executing the transaction.

        10. base_and_all_options (dollar):
            - The total contract value including base + option periods.

        11. base_exercised_options (dollar):
            - Contract value for the base period plus exercised options.

        12. cage_code (string):
            - Commercial and Government Entity (CAGE) code (5-character identifier).

        13. commercial_item_acquisition_standards (string):
            - Procedures used for acquiring commercial items.

        14. congressional_code (string):
            - U.S. Congressional District of the place of performance.

        15. contracting_offivers_determination_of_business_size (string):
            - The business size determination by the contracting officer (e.g. Small Business).

        16. cost_or_pricing_data (string):
            - Description indicating whether cost or pricing data was obtained or waived.

        17. cost_or_pricing_data_code (string):
            - Code for whether cost or pricing data was obtained or waived.

        18. country_code (string):
            - ISO country code of the recipient's location.

        19. country_name (string):
            - Name of the country corresponding to the country_code.

        20. date_signed (ISO date):
            - The date the contract was officially signed.

        21. description (string):
            - A textual description of the contract requirement.

        22. dod_acquisition_program_code (string):
            - Code used by DoD to identify the program or weapons system purchased.

        23. dod_acquisition_program_description (string):
            - Description of the DoD acquisition program code.

        24. end_date (ISO date):
            - End date (or completion date) of the contract's period of performance.

        25. extent_compete_description (string):
            - Description of the code in `extent_competed`.

        26. extent_competed (string):
            - Code representing how the contract was competed (e.g., full and open).

        27. fair_opportunity_limited_sources (string):
            - Explanation of limited competition if fair opportunity was not provided.

        28. funding_agency_name (string):
            - The name of the top-level department providing the majority of the funds.

        29. funding_agency_office_name (string):
            - The "level n" office providing the majority of the funds.

        30. funding_agency_subtier_agency_code (string):
            - Code for the sub-tier agency that provided the funds.

        31. funding_agency_technomile_id (int):
            - The TechnoMile GovSearch ID for the funding agency.

        32. funding_agency_technomile_name (string):
            - The TechnoMile GovSearch name for the funding agency.

        33. funding_agency_toptier_agency_code (string):
            - Code for the top-tier agency providing the majority of the funds.

        34. funding_office_code (string):
            - Code for the office that provided the majority of the funds.

        35. funding_sub_agency_name (string):
            - The name of the sub-tier agency that provided the funds.

        36. generated_unique_award_id (string):
            - A derived unique key for the prime award (concatenation of IDs).

        37. information_technology_commercial_item_category_code (string):
            - Code designating the commercial availability of an IT product/service.

        38. labor_standards (string):
            - Indicates if the contract is subject to labor standards (e.g. Service Contract Act).

        39. last_modified_date (ISO date):
            - The last date/time the record was changed in FPDS.

        40. location_country_code (string):
            - ISO code for the awardee's location country (e.g. "USA").

        41. multi_year_contract (string):
            - Description for the multi-year contract code.

        42. multi_year_contract_code (string):
            - Code indicating if it's a multi-year contract.

        43. naics (int):
            - 6-digit North American Industry Classification System code.

        44. naics_description (string):
            - Official NAICS title corresponding to the NAICS code.

        45. national_interest_action (string):
            - Description for the code in `national_interest_action_code`.

        46. national_interest_action_code (string):
            - Code representing a national interest action (e.g., emergency operation).

        47. number_of_actions (int):
            - The number input by the agency for how many actions are reported in one modification.

        48. number_of_offers_received (string):
            - The number of actual offers/bids in response to the solicitation.

        49. other_than_full_and_open_competition (string):
            - Description for "other than full and open competition."

        50. other_than_full_and_open_competition_code (string):
            - Code representing "other than full and open competition."

        51. parent_award_piid (string):
            - The parent IDV contract number if referencing an IDV.

        52. parent_award_single_or_multiple (string):
            - Description indicating single or multiple award IDV.

        53. parent_award_type (string):
            - Description of the code in `parent_award_type_code` (GWAC, BOA, BPA, etc.).

        54. parent_award_type_code (string):
            - The type of Indefinite Delivery Vehicle (e.g., GWAC, BOA, BPA).

        55. parent_recipient_name (string):
            - The name of the ultimate parent vendor/company.

        56. performance_based_service_acquisition (string):
            - Description indicating if the acquisition is performance-based.

        57. piid (string):
            - Unique ID for a task order.

        58. place_of_perf_country_desc (string):
            - The country name where the majority of performance occurs.

        59. primary_place_of_performance_city_name (string):
            - The city where the majority of the award's performance occurs.

        60. primary_place_of_performance_zip_4 (string):
            - ZIP+4 code for the primary place of performance.

        61. product_or_service_code (string):
            - 4-character code (PSC) identifying product/service purchased.

        62. product_or_service_code_description (string):
            - Official title describing the PSC.

        63. program_acronym (string):
            - Short name/title for a program (e.g., "SEWP," "ITOPS").

        64. recipient_name (string):
            - Name of the awardee/recipient.

        65. recipient_parent_uei (string):
            - Ultimate parent's UEI code.

        66. recipient_uei (string):
            - Awardee's UEI code (replacement for DUNS).

        67. recovered_materials_sustainability (string):
            - Indicates if recovered material or environmental clauses are included.

        68. solicitation_identifier (string):
            - ID linking transactions in FPDS to the solicitation.

        69. solicitation_procedure_description (string):
            - Description for the code in `solicitation_procedures`.

        70. solicitation_procedures (string):
            - Code for the competitive solicitation procedures used.

        71. start_date (ISO date):
            - Effective date or start date of the contract's period of performance.

        72. state_code (string):
            - USPS two-letter abbreviation of the state where the awardee is located.

        73. state_name (string):
            - The full state/territory name.

        74. subcontracting_plan (string):
            - Indicates if a subcontracting plan is required (FAR Part 19.702).

        75. total_obligation (dollar):
            - The total amount of money obligated for this award.

        76. type (string):
            - Broad category of contract action (FPDS: `contractActionType`).

        77. type_description (string):
            - Description for the `type` field (contract action type).

        78. type_of_contract_pricing (string):
            - Description of the contract pricing arrangement (e.g., Fixed-Price).

        79. type_of_contract_pricing_code (string):
            - Code identifying the type of contract pricing.

        80. type_of_set_aside (string):
            - Description of the set-aside category (e.g., SDVOSB, 8(a)).

        81. type_of_set_aside_code (string):
            - Code representing the set-aside type.
         
        RESPONSE GUIDELINES:
        - Base all answers exclusively on data from our GovSearchAI database
        - When analyzing contracts:
          * Compare values, dates, and patterns across similar contracts in our database
          * Look for relationships between vendors, agencies, and contract types
          * Use our historical data to identify trends
          * Connect related information across different contracts
        
        - For limited or unclear data:
          * Check similar contracts in our database
          * Look for patterns in vendor history within our system
          * Analyze agency contracting patterns from our records
          * Use available dates and values to establish timelines
        
        - Provide specific details from our database:
          * Exact dates, values, and numbers
          * Vendor relationships and contract patterns
          * Agency contracting history
          * Related contracts and follow-on work
        
        Never suggest external sources or databases. If information is limited, focus on analyzing available patterns and relationships within our existing database.
        
        Context for this query: {context}"""),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    chain = prompt | ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0,
        max_tokens=1000
    )
    
    response = chain.invoke({
        "messages": state["messages"],
        "context": state["context"]
    })
    
    return {
        "messages": [*state["messages"], response],
        "context": state["context"]
    }

def create_graph():
    workflow = StateGraph(State)
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

chain = create_graph()

def chat(message: str, chat_history: list[BaseMessage] = None) -> list[BaseMessage]:
    if chat_history is None:
        chat_history = []
    
    # Clean and prepare the message
    message = message.strip()
    
    result = chain.invoke({
        "messages": [*chat_history, HumanMessage(content=message)],
        "context": ""
    })
    
    # Extract the latest response
    latest_response = result["messages"][-1]
    
    # Update chat history
    chat_history = [*chat_history, HumanMessage(content=message), latest_response]
    
    return chat_history

app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app)

# MongoDB setup
client = MongoClient(os.getenv('MONGODB_URI'))
db = client['TMdata']
chat_collection = db['chat_history']

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.json
    message = data.get('message')
    email = data.get('email')
    name = data.get('name')
    chat_history = data.get('chatHistory', [])
    
    try:
        # Use the existing chat function
        response = chat(message)
        
        # Save to MongoDB
        save_chat_message(message, response[-1].content, email, name)
        
        return jsonify({'response': response[-1].content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/end-chat', methods=['POST'])
def end_chat():
    data = request.json
    email = data.get('email')
    name = data.get('name')
    messages = data.get('messages', [])
    
    try:
        # Find existing chat document
        existing_chat = chat_collection.find_one({
            'email': email,
            'name': name
        })
        
        if existing_chat:
            # Append new messages to existing ones
            chat_collection.update_one(
                {'_id': existing_chat['_id']},
                {
                    '$push': {
                        'messages': {
                            '$each': messages
                        }
                    },
                    '$set': {
                        'last_updated': datetime.utcnow()
                    }
                }
            )
        else:
            # Create new document only if user doesn't exist
            chat_collection.insert_one({
                'email': email,
                'name': name,
                'messages': messages,
                'created_at': datetime.utcnow(),
                'last_updated': datetime.utcnow()
            })
            
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

def save_chat_message(message, response, email, name):
    """Save or update chat message in MongoDB."""
    try:
        client = MongoClient(os.getenv('MONGODB_URI'))
        db = client['TMdata']
        chats_collection = db['chat_history']
        
        # Try to find existing chat document for the user
        existing_chat = chats_collection.find_one({
            'email': email,
            'name': name
        })
        
        if existing_chat:
            # Update existing chat document
            chats_collection.update_one(
                {'_id': existing_chat['_id']},
                {
                    '$push': {
                        'messages': {
                            '$each': [
                                {'type': 'user', 'content': message},
                                {'type': 'ai', 'content': response}
                            ]
                        }
                    },
                    '$set': {'last_updated': datetime.now()}
                }
            )
        else:
            # Create new chat document for new user
            chats_collection.insert_one({
                'email': email,
                'name': name,
                'messages': [
                    {'type': 'user', 'content': message},
                    {'type': 'ai', 'content': response}
                ],
                'created_at': datetime.now(),
                'last_updated': datetime.now()
            })
        
        client.close()
        return True
        
    except Exception as e:
        print(f"Error saving chat message: {str(e)}")
        return False

if __name__ == '__main__':
    app.run(debug=True)
