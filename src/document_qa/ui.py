import streamlit as st
import os
import pandas as pd
from pathlib import Path
from document_qa.config import Settings
from document_qa.rag import RAGPipeline, NOT_FOUND
from document_qa.indexer import build_index
from document_qa.embeddings import SentenceTransformerEmbedder
from document_qa.vector_store import ChromaVectorStore
from document_qa.generation import GeminiGenerator, OpenAIGenerator, OllamaGenerator

# Page Configurations
st.set_page_config(
    page_title="Grounded Q&A Assistant Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Translucent color-coded relevance badge helper
def get_relevance_badge(distance: float) -> str:
    if distance <= 0.45:
        # Green (High)
        bg, border, text = "rgba(40, 167, 69, 0.12)", "rgba(40, 167, 69, 0.25)", "#2ebd59"
        label = "High Relevance"
    elif distance <= 0.70:
        # Blue (Moderate)
        bg, border, text = "rgba(23, 162, 184, 0.12)", "rgba(23, 162, 184, 0.25)", "#17a2b8"
        label = "Moderate Relevance"
    elif distance <= 0.85:
        # Orange (Low)
        bg, border, text = "rgba(255, 193, 7, 0.12)", "rgba(255, 193, 7, 0.25)", "#ffc107"
        label = "Low Relevance"
    else:
        # Red (Filtered)
        bg, border, text = "rgba(220, 53, 69, 0.12)", "rgba(220, 53, 69, 0.25)", "#dc3545"
        label = "Filtered / Gated"
        
    return f"""
    <span style="
        background-color: {bg};
        color: {text};
        border: 1px solid {border};
        padding: 0.2rem 0.65rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    ">{label} (distance: {distance:.3f})</span>
    """

# Initialize session state for chat history and settings
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "settings" not in st.session_state:
    st.session_state.settings = Settings.from_env()

if "needs_rebuild" not in st.session_state:
    st.session_state.needs_rebuild = False

settings = st.session_state.settings

# Cache embedder and vector store loading
@st.cache_resource
def get_cached_components(embedding_model: str, embedding_batch_size: int, index_dir_str: str, collection_name: str):
    embedder = SentenceTransformerEmbedder(embedding_model, embedding_batch_size)
    store = ChromaVectorStore(Path(index_dir_str), collection_name)
    return embedder, store

# Custom Premium CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #6366f1 0%, #14b8a6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
        margin-top: -1.5rem;
    }
    .main-subtitle {
        color: #71717a;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🔍 Grounded Q&A Assistant Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='main-subtitle'>Ask questions grounded strictly in your local knowledge base documents with inline citations and transparent retrieval audit.</p>", unsafe_allow_html=True)

# Try loading components
try:
    with st.spinner("Downloading/Loading SentenceTransformers embedding model and ChromaDB database... (This might take 1-2 minutes on first startup on Streamlit Cloud)"):
        embedder, store = get_cached_components(
            settings.embedding_model,
            settings.embedding_batch_size,
            str(settings.index_dir),
            settings.collection_name
        )
except Exception as exc:
    st.error(f"Error loading local SentenceTransformers embedder or ChromaDB: {exc}")
    st.stop()

# Sidebar UI
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/search.png", width=65)
    st.markdown("<h2 style='margin-top:0; font-weight:700;'>Grounded Q&A Pro</h2>", unsafe_allow_html=True)
    st.caption("Retrieval-Augmented Generation (RAG) Control Center")
    
    st.divider()
    
    st.markdown("### ⚙️ Engine Settings")
    st.markdown(f"**LLM Backend:** `{settings.llm_provider.upper()}`")
    if settings.llm_provider == "gemini":
        st.markdown(f"**Model:** `{settings.gemini_chat_model}`")
    elif settings.llm_provider == "openai":
        st.markdown(f"**Model:** `{settings.openai_chat_model}`")
    else:
        st.markdown(f"**Model:** `{settings.ollama_model}`")
    st.markdown(f"**Embedder:** `all-MiniLM-L6-v2`")

    st.divider()
    st.markdown("### 🧠 Retrieval Parameters")
    
    top_k = st.slider("Top K Chunks", min_value=1, max_value=10, value=settings.top_k)
    if top_k != settings.top_k:
        settings.top_k = top_k
        
    max_distance = st.slider("Max Relevance Distance (Lower is stricter)", min_value=0.0, max_value=1.5, value=settings.max_distance, step=0.05)
    if max_distance != settings.max_distance:
        settings.max_distance = max_distance

    # Advanced Settings
    with st.expander("🛠️ Advanced Indexing Settings"):
        chunk_size = st.number_input("Chunk Size (Chars)", min_value=200, max_value=5000, value=settings.chunk_size, step=100)
        if chunk_size != settings.chunk_size:
            settings.chunk_size = chunk_size
            st.session_state.needs_rebuild = True
            
        chunk_overlap = st.number_input("Chunk Overlap (Chars)", min_value=0, max_value=settings.chunk_size-1, value=settings.chunk_overlap, step=10)
        if chunk_overlap != settings.chunk_overlap:
            settings.chunk_overlap = chunk_overlap
            st.session_state.needs_rebuild = True

# Instantiate generator dynamically
generator = None
try:
    if settings.llm_provider == "gemini":
        generator = GeminiGenerator(settings.gemini_chat_model)
    elif settings.llm_provider == "openai":
        generator = OpenAIGenerator(settings.openai_chat_model)
    else:
        generator = OllamaGenerator(settings.ollama_base_url, settings.ollama_model)
except Exception as e:
    st.sidebar.warning(f"⚠️ Generator error: {e}")

if generator:
    pipeline = RAGPipeline(embedder, store, generator, settings.top_k, settings.max_distance)
else:
    pipeline = None

# Sidebar File Manager
with st.sidebar:
    st.divider()
    st.markdown("### 📂 Document Manager")
    
    def list_knowledge_docs(data_dir: Path) -> list[Path]:
        if not data_dir.exists():
            return []
        from document_qa.ingestion import SUPPORTED_SUFFIXES
        return sorted(p for p in data_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)

    docs = list_knowledge_docs(settings.data_dir)
    
    # Upload new file
    uploaded_files = st.file_uploader("Upload new document(s)", type=["pdf", "txt", "docx"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        any_uploaded = False
        for f in uploaded_files:
            target_path = settings.data_dir / f.name
            if not target_path.exists():
                target_path.write_bytes(f.read())
                any_uploaded = True
        if any_uploaded:
            st.toast("Document uploaded successfully!", icon="📄")
            st.session_state.needs_rebuild = True
            st.rerun()

    # Show list of docs with delete action
    if docs:
        st.markdown(f"**Indexed Documents ({len(docs)}):**")
        for doc in docs:
            col_name, col_del = st.columns([0.85, 0.15])
            icon = "📄" if doc.suffix.lower() == ".txt" else "📕" if doc.suffix.lower() == ".pdf" else "📝"
            col_name.markdown(f"<span style='font-size:0.85rem;'>{icon} {doc.name}</span>", unsafe_allow_html=True)
            if col_del.button("🗑️", key=f"del_{doc.name}", help=f"Delete {doc.name}"):
                try:
                    doc.unlink()
                    st.toast(f"Deleted {doc.name}", icon="🗑️")
                    st.session_state.needs_rebuild = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting file: {e}")
    else:
        st.info("No documents uploaded yet.")
        
    try:
        chunk_count = store.count()
        st.markdown(f"**Total Indexed Chunks:** `{chunk_count}`")
    except Exception:
        st.markdown("**Total Indexed Chunks:** `0` (Index empty)")
        
    st.divider()
    
    # Rebuild index banner
    if st.session_state.get("needs_rebuild", False):
        st.warning("⚠️ Changes detected. Rebuild required.")
        
    if st.button("🔄 Rebuild Search Index", use_container_width=True, type="primary" if st.session_state.needs_rebuild else "secondary"):
        with st.status("Rebuilding vector store index...", expanded=True) as status:
            try:
                st.write("Ingesting files and parsing contents...")
                doc_count, chunk_count = build_index(settings, embedder, store)
                status.update(label=f"Index rebuilt! {doc_count} docs, {chunk_count} chunks.", state="complete", expanded=False)
                st.toast("Search index successfully rebuilt!", icon="✅")
                st.session_state.needs_rebuild = False
                st.rerun()
            except Exception as e:
                status.update(label="Rebuild failed", state="error")
                st.error(f"Error rebuilding index: {e}")
                
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# Settings at top of chat
col_rewriter, col_gap = st.columns([0.4, 0.6])
enable_rewriter = col_rewriter.toggle(
    "Enable Context-Aware Chat History", 
    value=True, 
    help="Use the LLM to rewrite your follow-up questions to resolve pronouns/references from prior turns."
)

# Render Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🔍"):
        st.markdown(message["content"])
        
        # If assistant has sources, render them under the answer in a beautiful container
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            
            # Show rewritten query if present
            if "search_query" in message and message["search_query"] and message["search_query"] != message.get("original_query"):
                st.caption(f"🔄 *Context-Aware Search Formulation:* \"{message['search_query']}\"")
                
            # Layout: Sources and Analytics
            tab_sources, tab_analytics = st.tabs(["📚 Retrieved Source Excerpts", "📊 Retrieval Analytics"])
            
            with tab_sources:
                for index, src in enumerate(message["sources"], start=1):
                    badge_html = get_relevance_badge(src["distance"])
                    with st.expander(f"[S{index}] {src['source']} ({src['locator']})"):
                        st.markdown(badge_html, unsafe_allow_html=True)
                        st.markdown(f"<div style='margin-top:0.5rem; font-size:0.95rem; line-height:1.5;'>{src['text']}</div>", unsafe_allow_html=True)
                        
            with tab_analytics:
                distances = [src["distance"] for src in message["sources"]]
                similarities = [max(0.0, 1.0 - d) for d in distances]
                names = [f"[S{i}] {src['source']}" for i, src in enumerate(message["sources"], start=1)]
                
                df_chart = pd.DataFrame({
                    "Cosine Similarity": similarities
                }, index=names)
                
                st.markdown("**Cosine Similarity per Chunk:** (Higher is more relevant)")
                st.bar_chart(df_chart)

# Quick chips selection
st.markdown("<p style='font-size:0.9rem; font-weight:600; margin-bottom:0.4rem;'>💡 Try a sample question:</p>", unsafe_allow_html=True)
samples = [
    "Why is passivation important at the end of a space mission?",
    "Why can warm nights make an urban heatwave more dangerous?",
    "What is a break-glass account and how should it be protected?",
    "Why should seagrass projects address water quality before transplanting?",
    "Who won yesterday's football match?"
]

cols = st.columns(len(samples))
clicked_sample = None
for i, sample in enumerate(samples):
    if cols[i].button(sample.split("?")[0] + "?", key=f"sample_{i}", use_container_width=True):
        clicked_sample = sample

# Resolve user input
user_query = st.chat_input("Ask a question about the documents...")
if clicked_sample:
    user_query = clicked_sample

if user_query:
    # Display user query in chat
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)
    
    # Add to session history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # Prepare history for the rewriter if enabled
    history_to_pass = []
    if enable_rewriter:
        for msg in st.session_state.chat_history[:-1]:
            history_to_pass.append({"role": msg["role"], "content": msg["content"]})
            
    # Query the RAG Pipeline and display response
    with st.chat_message("assistant", avatar="🔍"):
        if not pipeline:
            st.error("RAG pipeline not initialized. Check LLM provider settings or API keys in the sidebar.")
        else:
            with st.spinner("Analyzing sources and generating cited answer..."):
                try:
                    answer = pipeline.ask(user_query, history=history_to_pass)
                    
                    # If query was rewritten, notify the user
                    if answer.search_query and answer.search_query != user_query:
                        st.info(f"🔄 **Context-Aware Query Formulation:** \"{answer.search_query}\"")
                        
                    # Render answer text
                    st.markdown(answer.text)
                    
                    # Format sources for history serialization
                    serialized_sources = []
                    for item in answer.sources:
                        serialized_sources.append({
                            "source": item.chunk.source,
                            "locator": item.chunk.locator,
                            "text": item.chunk.text,
                            "distance": item.distance
                        })
                    
                    # Display sources and analytics tabs
                    if serialized_sources:
                        tab_sources, tab_analytics = st.tabs(["📚 Retrieved Source Excerpts", "📊 Retrieval Analytics"])
                        
                        with tab_sources:
                            for index, src in enumerate(serialized_sources, start=1):
                                badge_html = get_relevance_badge(src["distance"])
                                with st.expander(f"[S{index}] {src['source']} ({src['locator']})"):
                                    st.markdown(badge_html, unsafe_allow_html=True)
                                    st.markdown(f"<div style='margin-top:0.5rem; font-size:0.95rem; line-height:1.5;'>{src['text']}</div>", unsafe_allow_html=True)
                                    
                        with tab_analytics:
                            distances = [src["distance"] for src in serialized_sources]
                            similarities = [max(0.0, 1.0 - d) for d in distances]
                            names = [f"[S{i}] {src['source']}" for i, src in enumerate(serialized_sources, start=1)]
                            
                            df_chart = pd.DataFrame({
                                "Cosine Similarity": similarities
                            }, index=names)
                            st.markdown("**Cosine Similarity per Chunk:** (Higher is more relevant)")
                            st.bar_chart(df_chart)
                                    
                    # Add assistant reply to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer.text,
                        "sources": serialized_sources,
                        "search_query": answer.search_query,
                        "original_query": user_query
                    })
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error querying backend: {e}")
