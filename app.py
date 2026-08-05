# import os
# import re
# import json
# import shutil
# import streamlit as st
# from typing import List, Optional
# from pydantic import BaseModel


# # ==============================================================================
# # UPDATED LANGCHAIN & LLM IMPORTS
# # ==============================================================================
# from langchain_groq import ChatGroq
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# from langchain.chains import create_stuff_documents_chain, create_retrieval_chain

import os
import re
import json
import streamlit as st
from typing import List, Optional
from pydantic import BaseModel

# Safe, stable imports across all LangChain versions
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
#--------

# Document Chain Imports (Fixed)
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.retrieval import create_retrieval_chain

# ... rest of your code follows below ...

# LangChain & AI Libraries
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Streamlit Page Setup
st.set_page_config(page_title="Sephora AI Brand Engine", layout="wide")

# ==============================================================================
# SIDEBAR - API KEYS CONFIGURATION
# ==============================================================================
st.sidebar.title("⚙️ API Configuration")

# Get keys from Streamlit secrets OR manual input
default_google = st.secrets.get("GOOGLE_API_KEY", "") if hasattr(st, "secrets") else ""
default_groq = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""

google_api_key = st.sidebar.text_input("GOOGLE_API_KEY", value=default_google, type="password")
groq_api_key = st.sidebar.text_input("GROQ_API_KEY", value=default_groq, type="password")

if not google_api_key or not groq_api_key:
    st.warning("⚠️ Please provide both Google API Key and Groq API Key in the sidebar to proceed.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize LLMs
llm_groq = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7, groq_api_key=groq_api_key)
llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=google_api_key)

# ==============================================================================
# DATA MODELS & PARSER
# ==============================================================================
class ProductInput(BaseModel):
    brand_name: str = "Sephora"
    product_name: str
    category: str
    target_audience: Optional[str]
    key_features: List[str]

def parse_agent_json(content: str) -> dict:
    try:
        clean_content = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', content).strip()
        start = clean_content.find('{')
        end = clean_content.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(clean_content[start:end])
        return json.loads(clean_content)
    except Exception:
        return {}

# ==============================================================================
# AGENTS
# ==============================================================================
def agent_1_brand_voice(input_data: ProductInput):
    prompt = PromptTemplate.from_template(
        """You are AGENT 1: Brand Voice Specialist for high-end luxury beauty brands like {brand_name}.
Product: {product_name} (Category: {category})
Key Features: {key_features}

Return JSON with keys: "brand_personality", "tone_of_voice", "core_values" (list), "luxury_taglines" (list)."""
    )
    res = (prompt | llm_groq).invoke({
        "brand_name": input_data.brand_name,
        "product_name": input_data.product_name,
        "category": input_data.category,
        "key_features": ", ".join(input_data.key_features)
    })
    return parse_agent_json(res.content)

def agent_2_ad_copy(input_data: ProductInput, tone: str):
    prompt = PromptTemplate.from_template(
        """You are AGENT 2: High-Converting Ad Copywriter for Sephora.
Product: {product_name}, Tone: {tone}, Features: {key_features}

Return JSON with keys: "headlines" (list), "instagram_caption", "facebook_ad", "linkedin_copy", "email_marketing" (dict), "call_to_action" (list)."""
    )
    res = (prompt | llm_groq).invoke({
        "product_name": input_data.product_name,
        "tone": tone,
        "key_features": ", ".join(input_data.key_features)
    })
    return parse_agent_json(res.content)

def agent_3_marketing_strategy(input_data: ProductInput):
    prompt = PromptTemplate.from_template(
        """You are AGENT 3: Chief Marketing Officer for Sephora.
Product: {product_name} ({category}), Audience: {target_audience}

Return JSON with keys: "overall_strategy", "target_audience_segmentation", "customer_persona" (dict), "unique_selling_points" (list)."""
    )
    res = (prompt | llm_groq).invoke({
        "product_name": input_data.product_name,
        "category": input_data.category,
        "target_audience": input_data.target_audience
    })
    return parse_agent_json(res.content)

def agent_4_hashtags(input_data: ProductInput):
    prompt = PromptTemplate.from_template(
        """You are AGENT 4: SEO Specialist for Sephora. Product: {product_name} ({category}).
Return JSON with keys: "trending_hashtags" (list), "niche_hashtags" (list), "seo_keywords" (list)."""
    )
    res = (prompt | llm_groq).invoke({"product_name": input_data.product_name, "category": input_data.category})
    return parse_agent_json(res.content)

def agent_5_logo_prompt(input_data: ProductInput):
    prompt = PromptTemplate.from_template(
        """You are AGENT 5: Art Director for Sephora sub-brand {product_name}.
Return JSON with keys: "minimalist_logo_prompt", "luxury_3d_emblem_prompt", "modern_vector_badge_prompt"."""
    )
    res = (prompt | llm_gemini).invoke({"product_name": input_data.product_name})
    return parse_agent_json(res.content)

def agent_6_image_prompt(input_data: ProductInput):
    prompt = PromptTemplate.from_template(
        """You are AGENT 6: Visual Director for Sephora. Product: {product_name} ({category}).
Return JSON with keys: "photorealistic_ad_prompt", "editorial_studio_prompt", "social_media_aesthetic_prompt"."""
    )
    res = (prompt | llm_gemini).invoke({"product_name": input_data.product_name, "category": input_data.category})
    return parse_agent_json(res.content)

# ==============================================================================
# STREAMLIT UI & INTERFACE
# ==============================================================================
st.title("💄 Sephora Brand Voice & Ad Copy Generator")

st.sidebar.header("Product Input")
brand_name = st.sidebar.text_input("Brand", value="Sephora")
product_name = st.sidebar.text_input("Product Name", value="Rouge Pur Couture Lipstick")
category = st.sidebar.selectbox("Category", ["Lips", "Skincare", "Complexion", "Eye Makeup", "Fragrance"])
target_audience = st.sidebar.text_input("Target Audience", value="Luxury beauty enthusiasts, Gen Z")
features_raw = st.sidebar.text_area("Key Features (comma-separated)", value="16-hour wear, Hydrating Hyaluronic Acid, Satin finish")
key_features = [f.strip() for f in features_raw.split(",") if f.strip()]

product = ProductInput(
    brand_name=brand_name,
    product_name=product_name,
    category=category,
    target_audience=target_audience,
    key_features=key_features
)

tabs = st.tabs(["🚀 Full Campaign", "🎤 Brand Voice", "✍️ Ad Copy", "🎯 Strategy", "🏷️ Hashtags & SEO", "🎨 Logo Prompts", "📸 Image Prompts", "📚 RAG Chat"])

with tabs[0]:
    if st.button("Generate Full Campaign"):
        with st.spinner("Orchestrating 6 AI Agents..."):
            bv = agent_1_brand_voice(product)
            ac = agent_2_ad_copy(product, bv.get("tone_of_voice", "Luxury"))
            ms = agent_3_marketing_strategy(product)
            hs = agent_4_hashtags(product)
            lp = agent_5_logo_prompt(product)
            ip = agent_6_image_prompt(product)
            
            st.subheader("Brand Voice")
            st.json(bv)
            st.subheader("Ad Copy")
            st.json(ac)
            st.subheader("Marketing Strategy")
            st.json(ms)
            st.subheader("Hashtags & SEO")
            st.json(hs)
            st.subheader("Logo Prompts")
            st.json(lp)
            st.subheader("Image Ad Prompts")
            st.json(ip)

with tabs[1]:
    if st.button("Generate Brand Voice Only"):
        st.json(agent_1_brand_voice(product))

with tabs[2]:
    if st.button("Generate Ad Copy Only"):
        st.json(agent_2_ad_copy(product, "Luxury"))

with tabs[3]:
    if st.button("Generate Strategy Only"):
        st.json(agent_3_marketing_strategy(product))

with tabs[4]:
    if st.button("Generate Hashtags & SEO"):
        st.json(agent_4_hashtags(product))

with tabs[5]:
    if st.button("Generate Logo Prompts"):
        st.json(agent_5_logo_prompt(product))

with tabs[6]:
    if st.button("Generate Image Prompts"):
        st.json(agent_6_image_prompt(product))

with tabs[7]:
    st.header("📚 RAG Document Chat")
    uploaded_file = st.file_uploader("Upload Sephora Brand PDF", type=["pdf"])
    
    if uploaded_file is not None:
        os.makedirs("./uploaded_docs", exist_ok=True)
        file_path = os.path.join("./uploaded_docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        with st.spinner("Processing and indexing document..."):
            loader = PyPDFLoader(file_path)
            docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(loader.load())
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
            st.session_state.vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)
            st.success("Document indexed successfully!")

    st.divider()
    user_query = st.text_input("Ask a question about the uploaded Sephora document:")
    
    if st.button("Ask RAG Agent") and user_query:
        if "vector_store" not in st.session_state:
            st.error("Please upload and index a PDF document first.")
        else:
            with st.spinner("Retrieving relevant context and generating answer..."):
                # 1. Retrieve relevant document chunks
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.invoke(user_query)
                
                # 2. Combine chunks into text context
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # 3. Direct RAG Prompt
                rag_prompt = PromptTemplate.from_template(
                    """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question accurately and concisely.
If you don't know the answer based on the document, state that clearly.

Context:
{context}

Question: {question}
Answer:"""
                )
                
                # 4. Invoke LLM chain via native LCEL pipe (|)
                rag_chain = rag_prompt | llm_groq
                response = rag_chain.invoke({"context": context_text, "question": user_query})
                
                # 5. Render Output
                st.markdown(f"### Answer:\n{response.content}")
                
                with st.expander("View Source Context Chunks"):
                    for idx, doc in enumerate(retrieved_docs, start=1):
                        st.info(f"**Chunk {idx}:**\n{doc.page_content}")

with tabs[8]:
    st.header("📚 RAG Document Chat")
    uploaded_file = st.file_uploader("Upload Sephora Brand PDF", type=["pdf"])
    
    if uploaded_file is not None:
        os.makedirs("./uploaded_docs", exist_ok=True)
        file_path = os.path.join("./uploaded_docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        with st.spinner("Processing and indexing document..."):
            loader = PyPDFLoader(file_path)
            docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(loader.load())
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
            st.session_state.vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)
            st.success("Document indexed successfully!")

    st.divider()
    user_query = st.text_input("Ask a question about the uploaded Sephora document:")
    
    if st.button("Ask RAG Agent") and user_query:
        if "vector_store" not in st.session_state:
            st.error("Please upload and index a PDF document first.")
        else:
            with st.spinner("Retrieving relevant context and generating answer..."):
                # 1. Retrieve relevant document chunks
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.invoke(user_query)
                
                # 2. Combine chunks into text context
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # 3. Direct RAG Prompt
                rag_prompt = PromptTemplate.from_template(
                    """You are an assistant for question-answering tasks.
                        Use the following pieces of retrieved context to answer the question accurately and concisely.
                        If you don't know the answer, state that you don't know based on the document.
                        
                        Context:
                        {context}
                        
                        Question: {question}
                        Answer:"""
                     )
                
                # 4. Invoke LLM chain via standard pipe (|)
                rag_chain = rag_prompt | llm_groq
                response = rag_chain.invoke({"context": context_text, "question": user_query})
                
                # 5. Render Output
                st.markdown(f"### Answer:\n{response.content}")
                
                with st.expander("View Source Context Chunks"):
                    for idx, doc in enumerate(retrieved_docs, start=1):
                        st.info(f"**Chunk {idx}:**\n{doc.page_content}")
