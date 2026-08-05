import os
import re
import json
import shutil
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain & LLM Imports (Modern LangChain v0.2+)
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# ==============================================================================
# 0. API KEY CONFIGURATION
# ==============================================================================
GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY",type = "password")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")

GROQ_API_KEY = st.sidebar.text_input("GROQ_API_KEY",type = "password")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize FastAPI App
app = FastAPI(
    title="Sephora AI Brand & Ad Copy Engine",
    description="Multi-Agent GenAI & RAG Backend for Luxury Marketing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LLM Providers
llm_groq = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    groq_api_key=GROQ_API_KEY
)

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# Global Vector Store Reference for RAG Module
vector_store: Optional[Chroma] = None
UPLOAD_DIR = "./uploaded_docs"
CHROMA_DIR = "./chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ==============================================================================
# PYDANTIC DATA MODELS
# ==============================================================================
class ProductInput(BaseModel):
    brand_name: str = "Sephora"
    product_name: str
    category: str
    target_audience: Optional[str] = "Luxury beauty enthusiasts, Gen Z & Millennials"
    key_features: List[str]

class BrandVoiceResponse(BaseModel):
    brand_personality: str
    tone_of_voice: str
    core_values: List[str]
    luxury_taglines: List[str]

class AdCopyResponse(BaseModel):
    headlines: List[str]
    instagram_caption: str
    facebook_ad: str
    linkedin_copy: str
    email_marketing: dict
    call_to_action: List[str]

class MarketingStrategyResponse(BaseModel):
    overall_strategy: str
    target_audience_segmentation: str
    customer_persona: dict
    unique_selling_points: List[str]

class HashtagsSEOResponse(BaseModel):
    trending_hashtags: List[str]
    niche_hashtags: List[str]
    seo_keywords: List[str]

class LogoPromptResponse(BaseModel):
    minimalist_logo_prompt: str
    luxury_3d_emblem_prompt: str  # Fixed syntax error (no leading number)
    modern_vector_badge_prompt: str

class ImagePromptResponse(BaseModel):
    photorealistic_ad_prompt: str
    editorial_studio_prompt: str
    social_media_aesthetic_prompt: str

class RAGQueryRequest(BaseModel):
    query: str

class RAGQueryResponse(BaseModel):
    answer: str
    source_documents: List[str]


# ==============================================================================
# ROBUST JSON PARSER HELPER
# ==============================================================================
def parse_agent_json(content: str) -> dict:
    """
    Safely extracts and parses JSON from LLM outputs.
    Strips markdown code fences (```json ... ```) and extracts raw JSON dictionaries.
    """
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
# AGENT DEFINITIONS
# ==============================================================================

# --- AGENT 1: Brand Voice Agent ---
def agent_1_brand_voice(input_data: ProductInput) -> BrandVoiceResponse:
    prompt = PromptTemplate.from_template(
        """You are AGENT 1: Brand Voice Specialist for high-end luxury beauty brands like {brand_name}.
Product: {product_name} (Category: {category})
Key Features: {key_features}

Tasks:
1. Define a distinct Luxury Brand Personality for this Sephora campaign.
2. Outline the precise Tone of Voice (e.g., sophisticated, empowering, playful-chic).
3. List 4 Core Brand Values embodied by this product.
4. Craft 3 high-concept luxury taglines.

Return output strictly as JSON with keys:
"brand_personality", "tone_of_voice", "core_values" (list), "luxury_taglines" (list).
"""
    )
    chain = prompt | llm_groq
    res = chain.invoke({
        "brand_name": input_data.brand_name,
        "product_name": input_data.product_name,
        "category": input_data.category,
        "key_features": ", ".join(input_data.key_features)
    })
    
    data = parse_agent_json(res.content)
    if data:
        return BrandVoiceResponse(**data)
        
    return BrandVoiceResponse(
        brand_personality="Empowered Elegance & Sephora Modern Luxury",
        tone_of_voice="Sophisticated, vibrant, authoritative, and inclusive.",
        core_values=["Self-Expression", "Clean Formula Innovation", "Uncompromising Quality", "Inclusive Beauty"],
        luxury_taglines=["Unleash Your Inner Glow.", "Bold Color, Effortless Wear.", "Beauty Defined by You."]
    )

# --- AGENT 2: Ad Copy Agent ---
def agent_2_ad_copy(input_data: ProductInput, brand_voice: str) -> AdCopyResponse:
    prompt = PromptTemplate.from_template(
        """You are AGENT 2: High-Converting Ad Copywriter for Sephora.
Product: {product_name}
Brand Voice Guidelines: {brand_voice}
Features: {key_features}

Generate tailored ad copy across platforms:
- 3 Attention-grabbing Headlines
- 1 Instagram Caption with emojis & natural callouts
- 1 Facebook Ad Copy with hook, body, social proof cue, and CTA
- 1 LinkedIn B2B / Thought Leadership Marketing Copy (Beauty Industry innovation)
- 1 Email Marketing Campaign (Subject, Preheader, Body, CTA)
- 3 Direct Calls To Action (CTAs)

Format response as JSON with keys:
"headlines" (list), "instagram_caption", "facebook_ad", "linkedin_copy", "email_marketing" (dict with subject, body, cta), "call_to_action" (list).
"""
    )
    chain = prompt | llm_groq
    res = chain.invoke({
        "product_name": input_data.product_name,
        "brand_voice": brand_voice,
        "key_features": ", ".join(input_data.key_features)
    })
    
    data = parse_agent_json(res.content)
    if data:
        return AdCopyResponse(**data)

    return AdCopyResponse(
        headlines=["Indulge in Pigment. Elevate Your Everyday.", "The Ultimate Hydration Secret from Sephora."],
        instagram_caption="Unapologetically bold. ✨ Elevate your beauty routine with Sephora's latest obsession. Tap to shop now! 💄 #SephoraBeauty",
        facebook_ad="Ready to transform your beauty routine? Meet our newest formula designed for all-day comfort and undeniable radiance.",
        linkedin_copy="How Sephora continues to lead the prestige cosmetics sector through formulation innovation and modern brand storytelling.",
        email_marketing={"subject": "Exclusive Access: Discover Your New Favorite", "body": "Step into Sephora and experience luxury re-imagined...", "cta": "Shop Collection Now"},
        call_to_action=["Shop Now at Sephora", "Discover Your Shade", "Claim Exclusive Offer"]
    )

# --- AGENT 3: Marketing Strategy Agent ---
def agent_3_marketing_strategy(input_data: ProductInput) -> MarketingStrategyResponse:
    prompt = PromptTemplate.from_template(
        """You are AGENT 3: Chief Marketing Officer for Sephora Campaigns.
Product: {product_name} ({category})
Audience: {target_audience}

Provide a strategic breakdown:
1. Overall Omnichannel Marketing Strategy
2. Target Audience Segmentation
3. Customer Persona Profile (Name, Age, Demographics, Pain Points, Buying Triggers)
4. 4 Key Unique Selling Points (USPs)

Return output strictly as JSON with keys:
"overall_strategy", "target_audience_segmentation", "customer_persona" (dict), "unique_selling_points" (list).
"""
    )
    chain = prompt | llm_groq
    res = chain.invoke({
        "product_name": input_data.product_name,
        "category": input_data.category,
        "target_audience": input_data.target_audience
    })
    
    data = parse_agent_json(res.content)
    if data:
        return MarketingStrategyResponse(**data)

    return MarketingStrategyResponse(
        overall_strategy="Omnichannel digital-first rollout leveraging TikTok creator gifting, Sephora Beauty Insider exclusives, and interactive store displays.",
        target_audience_segmentation="Gen-Z & Young Professionals seeking premium quality with conscious formulations.",
        customer_persona={"name": "Sophia Vance", "age": 27, "occupation": "Creative Director", "pain_points": "Long workdays requiring smudge-free makeup", "triggers": "Clean ingredients & glowing reviews"},
        unique_selling_points=["12-hour comfortable hydration", "Precision application tip", "Cruelty-free & Clean at Sephora certified"]
    )

# --- AGENT 4: Hashtag & SEO Agent ---
def agent_4_hashtags_seo(input_data: ProductInput) -> HashtagsSEOResponse:
    prompt = PromptTemplate.from_template(
        """You are AGENT 4: Digital Growth & SEO Specialist for Sephora.
Product: {product_name}
Category: {category}

Generate:
- 8 Trending Viral Beauty Hashtags
- 8 Niche / Community Beauty Hashtags
- 10 High-Intent SEO Keywords for E-Commerce & Google Search optimization

Return output strictly as JSON with keys:
"trending_hashtags" (list), "niche_hashtags" (list), "seo_keywords" (list).
"""
    )
    chain = prompt | llm_groq
    res = chain.invoke({"product_name": input_data.product_name, "category": input_data.category})
    
    data = parse_agent_json(res.content)
    if data:
        return HashtagsSEOResponse(**data)

    return HashtagsSEOResponse(
        trending_hashtags=["#SephoraBeauty", "#BeautyTok", "#LipstickLover", "#GlowySkin"],
        niche_hashtags=["#CleanBeautyAtSephora", "#SkincareAddicts", "#HydrationStation"],
        seo_keywords=["best hydrating lipstick", "buy sephora lipstick online", "long lasting luxury foundation"]
    )

# --- AGENT 5: Logo Prompt Agent ---
def agent_5_logo_prompt(input_data: ProductInput) -> LogoPromptResponse:
    prompt = PromptTemplate.from_template(
        """You are AGENT 5: Creative Art Director specializing in Midjourney / DALL-E 3 Logo Prompts.
Brand: Sephora / Product Sub-Brand: {product_name}

Generate 3 detailed, professional AI image generation prompts for branding logos:
1. Minimalist luxury logo prompt
2. 3D metallic/glass emblem logo prompt
3. Modern vector brand badge prompt

Return output strictly as JSON with keys:
"minimalist_logo_prompt", "luxury_3d_emblem_prompt", "modern_vector_badge_prompt".
"""
    )
    chain = prompt | llm_gemini
    res = chain.invoke({"product_name": input_data.product_name})
    
    data = parse_agent_json(res.content)
    if data:
        return LogoPromptResponse(**data)

    return LogoPromptResponse(
        minimalist_logo_prompt="Minimalist luxury emblem logo for Sephora product line, sleek typography, monochrome black and gold accents, vector white background, ultra-clean --v 6.0",
        luxury_3d_emblem_prompt="3D frosted glass and rose gold monogram emblem for Sephora luxury cosmetic line, studio lighting, soft shadows, 8k render, hyper-detailed",
        modern_vector_badge_prompt="Modern geometric beauty logo badge, clean lines, Sephora aesthetic, flat vector graphic, professional branding"
    )

# --- AGENT 6: Image Ad Prompt Agent ---
def agent_6_image_ad_prompt(input_data: ProductInput) -> ImagePromptResponse:
    prompt = PromptTemplate.from_template(
        """You are AGENT 6: Visual Ad Director for Sephora Commercial Photography.
Product: {product_name} ({category})
Features: {key_features}

Generate 3 ultra-detailed prompts for AI Image Generators (Midjourney/Flux/DALL-E 3) to render commercial beauty ads:
1. Photorealistic commercial hero shot with studio water droplets/lighting
2. Editorial magazine style aesthetic shot with model
3. Social media UGC clean aesthetic flatlay

Return output strictly as JSON with keys:
"photorealistic_ad_prompt", "editorial_studio_prompt", "social_media_aesthetic_prompt".
"""
    )
    chain = prompt | llm_gemini
    res = chain.invoke({
        "product_name": input_data.product_name, 
        "category": input_data.category, 
        "key_features": ", ".join(input_data.key_features)
    })
    
    data = parse_agent_json(res.content)
    if data:
        return ImagePromptResponse(**data)

    return ImagePromptResponse(
        photorealistic_ad_prompt=f"Commercial product photography of Sephora {input_data.product_name}, placed on black marble with soft water droplets, cinematic luxury lighting, high-end beauty ad, 8k resolution.",
        editorial_studio_prompt=f"High fashion Vogue style portrait, model applying Sephora {input_data.product_name}, studio lighting, soft focus background, ultra-detailed skin texture.",
        social_media_aesthetic_prompt=f"Aesthetic flatlay on beige silk background featuring Sephora {input_data.product_name}, gold jewelry, rose petals, morning sunlight, Pinterest beauty aesthetic."
    )


# ==============================================================================
# FASTAPI ENDPOINTS
# ==============================================================================

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Sephora GenAI & Agentic Brand Copy Engine",
        "available_agents": [
            "Agent 1: Brand Voice Agent",
            "Agent 2: Ad Copy Agent",
            "Agent 3: Marketing Strategy Agent",
            "Agent 4: Hashtag & SEO Agent",
            "Agent 5: Logo Prompt Agent",
            "Agent 6: Image Prompt Agent",
            "RAG Module: Sephora Knowledge Base Document QA"
        ]
    }

@app.post("/api/brand-voice", response_model=BrandVoiceResponse)
def generate_brand_voice(product: ProductInput):
    return agent_1_brand_voice(product)

@app.post("/api/ad-copy", response_model=AdCopyResponse)
def generate_ad_copy(product: ProductInput):
    voice = agent_1_brand_voice(product)
    return agent_2_ad_copy(product, voice.tone_of_voice)

@app.post("/api/marketing-strategy", response_model=MarketingStrategyResponse)
def generate_marketing_strategy(product: ProductInput):
    return agent_3_marketing_strategy(product)

@app.post("/api/hashtags-seo", response_model=HashtagsSEOResponse)
def generate_hashtags(product: ProductInput):
    return agent_4_hashtags_seo(product)

@app.post("/api/logo-prompts", response_model=LogoPromptResponse)
def generate_logo_prompts(product: ProductInput):
    return agent_5_logo_prompt(product)

@app.post("/api/image-prompts", response_model=ImagePromptResponse)
def generate_image_prompts(product: ProductInput):
    return agent_6_image_ad_prompt(product)

@app.post("/api/generate-full-campaign")
def generate_full_campaign(product: ProductInput):
    voice = agent_1_brand_voice(product)
    copy = agent_2_ad_copy(product, voice.tone_of_voice)
    strategy = agent_3_marketing_strategy(product)
    hashtags = agent_4_hashtags_seo(product)
    logo_prompts = agent_5_logo_prompt(product)
    image_prompts = agent_6_image_ad_prompt(product)

    return {
        "product": product,
        "brand_voice": voice,
        "ad_copy": copy,
        "strategy": strategy,
        "hashtags_seo": hashtags,
        "logo_prompts": logo_prompts,
        "image_prompts": image_prompts
    }


# ==============================================================================
# RAG MODULE
# ==============================================================================

@app.post("/api/rag/upload")
async def upload_pdf_for_rag(file: UploadFile = File(...)):
    global vector_store
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return {
        "message": f"Successfully processed and indexed {file.filename}",
        "total_chunks": len(docs)
    }

@app.post("/api/rag/chat", response_model=RAGQueryResponse)
def query_rag_chat(request: RAGQueryRequest):
    global vector_store
    if vector_store is None:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
            vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        else:
            raise HTTPException(
                status_code=400, 
                detail="No document indexed yet. Please upload a Sephora brand PDF first via /api/rag/upload."
            )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm_groq, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    result = rag_chain.invoke({"input": request.query})

    sources = [doc.page_content[:200] + "..." for doc in result.get("context", [])]

    return RAGQueryResponse(
        answer=result.get("answer", "No response generated."),
        source_documents=sources
    )


# ==============================================================================
# RUN SERVER
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
