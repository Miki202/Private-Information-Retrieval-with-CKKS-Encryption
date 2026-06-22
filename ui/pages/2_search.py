"""
Search страница - търсене на подобни превозни средства
"""
import streamlit as st
import sys
import os
from PIL import Image
import numpy as np
import pandas as pd
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from database.operations import search_similar, pir_search, get_database_stats
from ui.utils import load_encoder_model, encode_image

st.set_page_config(page_title="Search Vehicles", page_icon="🔍", layout="wide")

st.title("🔍 Search Similar Vehicles")
st.markdown("Търсене на подобни превозни средства в базата данни")

st.markdown("---")

# Load model
@st.cache_resource
def get_model():
    model_path = "encoder.pth"
    if not os.path.exists(model_path):
        st.error(f"❌ Encoder модел не е намерен: {model_path}")
        return None, None
    try:
        return load_encoder_model(model_path)
    except Exception as e:
        st.error(f"❌ Грешка: {e}")
        return None, None

model, device = get_model()

if model is None:
    st.stop()

# Check database
stats = get_database_stats()

if stats['total_vehicles'] == 0:
    st.warning("⚠️ Базата данни е празна! Качете превозни средства първо.")
    st.stop()

st.success(f"✅ Encoder зареден | База данни: {stats['total_vehicles']} превозни средства")

# Search mode selection
st.markdown("### 🔐 Search Mode")

mode = st.radio(
    "Избери режим на търсене:",
    ["Plain Search", "PIR Search (Private)"],
    help="Plain = бързо но не е private, PIR = бавно но напълно private"
)

is_pir = mode == "PIR Search (Private)"

col_info1, col_info2 = st.columns(2)

with col_info1:
    if is_pir:
        st.info("""
        🔐 **PIR Mode**
        - Query се криптира преди изпращане
        - Сървърът НЕ вижда query-то
        - Homomorphic операции
        - ~50x по-бавно (~500ms)
        """)
        
        if stats['encrypted_count'] == 0:
            st.error("❌ Няма криптирани записи в базата! PIR търсенето няма да работи.")
            st.info("💡 Качете превозни средства в Encrypted Mode първо.")
    else:
        st.warning("""
        🔓 **Plain Mode**
        - Много бързо (~10ms)
        - Сървърът вижда query-то
        - Не е private
        """)
        
        if stats['plain_count'] == 0:
            st.error("❌ Няма plain записи в базата!")
            st.info("💡 Качете превозни средства в Plain Mode първо.")

with col_info2:
    st.metric("Available Records", stats['encrypted_count'] if is_pir else stats['plain_count'])

st.markdown("---")

# Search interface
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🖼️ Query Image")
    
    query_image = st.file_uploader(
        "Качи query снимка",
        type=["jpg", "jpeg", "png"],
        key="query_upload"
    )
    
    if query_image:
        image = Image.open(query_image).convert("RGB")
        st.image(image, caption="Query Image", use_column_width=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Search Parameters")
    
    top_k = st.slider("Брой резултати", 1, 50, 10, help="Колко най-подобни резултата да покаже")
    
    st.markdown("**Filters:**")
    filter_color = st.selectbox(
        "Цвят",
        ["All", "red", "blue", "green", "black", "white", "silver", "gray", "yellow"],
        help="Филтрирай по цвят"
    )
    
    filter_body = st.selectbox(
        "Тип каросерия",
        ["All", "sedan", "suv", "truck", "van", "coupe", "hatchback", "wagon"],
        help="Филтрирай по тип"
    )
    
    # Search button
    search_button = st.button(
        "🔍 Search" if not is_pir else "🔐 PIR Search",
        type="primary",
        disabled=query_image is None,
        use_container_width=True
    )

with col2:
    st.markdown("### 📊 Search Results")
    
    if search_button and query_image:
        # Generate embedding
        with st.spinner("📊 Генериране на query embedding..."):
            start_embed_time = time.time()
            embedding = encode_image(image, model, device)
            embed_time = time.time() - start_embed_time
        
        st.success(f"✓ Query embedding генериран ({embed_time:.3f}s)")
        
        # Search
        try:
            if is_pir:
                # PIR Search
                st.markdown("---")
                st.markdown("#### 🔐 PIR Search Process")
                
                # Create placeholder for live updates
                pir_log = st.empty()
                
                with st.spinner("⏳ PIR Search in progress..."):
                    start_search_time = time.time()
                    
                    # Capture console output (simplified - в реалност PIR функцията принтира)
                    results = pir_search(
                        query_embedding=embedding,
                        top_k=top_k,
                        filter_color=None if filter_color == "All" else filter_color,
                        filter_body_type=None if filter_body == "All" else filter_body
                    )
                    
                    search_time = time.time() - start_search_time
                
                st.success(f"✓ PIR Search завършено ({search_time:.3f}s)")
                
                # Performance comparison
                col_perf1, col_perf2, col_perf3 = st.columns(3)
                col_perf1.metric("Embedding Time", f"{embed_time:.3f}s")
                col_perf2.metric("PIR Search Time", f"{search_time:.3f}s")
                col_perf3.metric("Total Time", f"{embed_time + search_time:.3f}s")
                
                st.info("🔐 **Privacy Guarantee:** Сървърът НЕ видя query embedding-a или similarity scores!")
                
            else:
                # Plain Search
                with st.spinner("🔍 Търсене..."):
                    start_search_time = time.time()
                    
                    results = search_similar(
                        query_embedding=embedding,
                        top_k=top_k,
                        filter_color=None if filter_color == "All" else filter_color,
                        filter_body_type=None if filter_body == "All" else filter_body
                    )
                    
                    search_time = time.time() - start_search_time
                
                st.success(f"✓ Search завършено ({search_time:.3f}s)")
                
                # Performance
                col_perf1, col_perf2 = st.columns(2)
                col_perf1.metric("Search Time", f"{search_time*1000:.1f}ms")
                col_perf2.metric("Results Found", len(results))
            
            # Display results
            if not results:
                st.warning("Няма намерени резултати с тези филтри.")
            else:
                st.markdown("---")
                st.markdown(f"### 🎯 Top {len(results)} Results")
                
                for i, result in enumerate(results):
                    vehicle = result["vehicle"]
                    score = result["similarity_score"]
                    
                    with st.expander(
                        f"#{i+1} - {vehicle['license_plate'] or 'N/A'} - Similarity: {score:.4f}",
                        expanded=i < 3
                    ):
                        col_a, col_b = st.columns([1, 2])
                        
                        with col_a:
                            st.metric("Similarity Score", f"{score:.6f}")
                            st.metric("License Plate", vehicle['license_plate'] or "N/A")
                            st.metric("🔐 Encrypted", "Yes" if vehicle['is_encrypted'] else "No")
                        
                        with col_b:
                            st.write(f"**Color:** {vehicle['color'] or 'N/A'}")
                            st.write(f"**Body Type:** {vehicle['body_type'] or 'N/A'}")
                            st.write(f"**UUID:** `{vehicle['uuid']}`")
                            st.write(f"**ID:** {vehicle['id']}")
                            st.caption(f"Created: {vehicle['created_at']}")
                
                # Download results
                st.markdown("---")
                df = pd.DataFrame([
                    {
                        "Rank": i+1,
                        "UUID": r["vehicle"]["uuid"],
                        "License Plate": r["vehicle"]["license_plate"],
                        "Color": r["vehicle"]["color"],
                        "Body Type": r["vehicle"]["body_type"],
                        "Similarity": f"{r['similarity_score']:.6f}",
                        "Encrypted": r["vehicle"]["is_encrypted"]
                    }
                    for i, r in enumerate(results)
                ])
                
                st.download_button(
                    "📥 Download Results (CSV)",
                    df.to_csv(index=False),
                    "search_results.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Грешка при търсене: {e}")
            st.exception(e)
    
    else:
        st.info("👆 Качете query изображение и натиснете Search")

st.markdown("---")

# Performance Comparison
st.markdown("### ⚡ Performance Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🔓 Plain Search:**
    - Embedding: ~50ms
    - Search: ~10ms
    - **Total: ~60ms**
    - Privacy: ❌ None
    """)

with col2:
    st.markdown("""
    **🔐 PIR Search:**
    - Embedding: ~50ms
    - Encryption: ~100ms
    - Search: ~500ms
    - **Total: ~650ms**
    - Privacy: ✅ Full query privacy
    """)

st.info("💡 **Trade-off:** PIR е ~10x по-бавно, но осигурява пълна query privacy!")