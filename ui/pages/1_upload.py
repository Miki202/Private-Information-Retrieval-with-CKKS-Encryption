"""
Upload страница - качване на превозни средства
"""
import streamlit as st
import sys
import os
from PIL import Image
import numpy as np
import time

# Добавяме parent директория
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from database.operations import insert_vehicle, insert_vehicle_encrypted
from ui.utils import load_encoder_model, encode_image, format_bytes

st.set_page_config(page_title="Upload Vehicle", page_icon="📤", layout="wide")

st.title("📤 Upload Vehicle")
st.markdown("Качване на ново превозно средство в базата данни")

st.markdown("---")

# Load encoder model (cached)
@st.cache_resource
def get_model():
    """Зарежда encoder модела (cache-ва се)"""
    model_path = "encoder.pth"  # Промени пътя според нуждите
    
    # Проверка дали файлът съществува
    if not os.path.exists(model_path):
        st.error(f"❌ Encoder модел не е намерен: {model_path}")
        st.info("💡 Моля постави encoder.pth файла в storage/ директорията")
        return None, None
    
    try:
        return load_encoder_model(model_path)
    except Exception as e:
        st.error(f"❌ Грешка при зареждане на модела: {e}")
        return None, None

model, device = get_model()

if model is None:
    st.stop()

st.success("✅ Encoder модел зареден успешно")

# Mode selection
st.markdown("### 🔐 Upload Mode")

mode = st.radio(
    "Избери режим на качване:",
    ["Plain Mode", "Encrypted Mode (PIR)"],
    help="Plain = бързо, Encrypted = PIR-ready с криптиране"
)

is_encrypted = mode == "Encrypted Mode (PIR)"

if is_encrypted:
    st.info("🔐 **Encrypted Mode**: Embedding-ът ще се криптира с CKKS преди съхраняване. PIR-ready!")
else:
    st.warning("🔓 **Plain Mode**: Embedding-ът ще се съхрани некриптиран. По-бързо, но не е private.")

st.markdown("---")

# Upload form
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🖼️ Vehicle Image")
    
    uploaded_file = st.file_uploader(
        "Качи снимка на превозно средство",
        type=["jpg", "jpeg", "png"],
        help="Поддържани формати: JPG, PNG"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Качена снимка", use_column_width=True)
        
        # Image info
        st.caption(f"Размер: {image.size[0]} × {image.size[1]} pixels")
        st.caption(f"Файл: {uploaded_file.name}")

with col2:
    st.markdown("### 📝 Vehicle Metadata")
    
    st.markdown("**Основна информация:**")
    license_plate = st.text_input(
        "Номер на колата",
        placeholder="ABC123",
        help="Регистрационен номер"
    )
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        color = st.selectbox(
            "Цвят",
            ["", "red", "blue", "green", "black", "white", "silver", "gray", "yellow", "other"],
            help="Изберете цвят на колата"
        )
    
    with col_b:
        body_type = st.selectbox(
            "Тип каросерия",
            ["", "sedan", "suv", "truck", "van", "coupe", "hatchback", "wagon", "other"],
            help="Изберете тип на каросерията"
        )
    
    st.markdown("---")
    
    # Upload button
    upload_button = st.button(
        "🚀 Upload Vehicle",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True
    )
    
    if upload_button and uploaded_file:
        with st.spinner("⏳ Обработка..."):
            try:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Стъпка 1: Генериране на embedding
                status_text.text("📊 Генериране на embedding...")
                progress_bar.progress(20)
                
                start_time = time.time()
                embedding = encode_image(image, model, device)
                embedding_time = time.time() - start_time
                
                status_text.text(f"✓ Embedding генериран ({embedding_time:.3f}s)")
                progress_bar.progress(40)
                
                # Стъпка 2: Вмъкване в базата данни
                if is_encrypted:
                    status_text.text("🔐 Криптиране и съхраняване...")
                    progress_bar.progress(60)
                    
                    start_time = time.time()
                    vehicle_uuid, vehicle_id = insert_vehicle_encrypted(
                        license_plate=license_plate if license_plate else None,
                        color=color if color else None,
                        body_type=body_type if body_type else None,
                        embedding=embedding,
                        image_path=uploaded_file.name
                    )
                    upload_time = time.time() - start_time
                    
                else:
                    status_text.text("💾 Съхраняване...")
                    progress_bar.progress(60)
                    
                    start_time = time.time()
                    vehicle_uuid, vehicle_id = insert_vehicle(
                        license_plate=license_plate if license_plate else None,
                        color=color if color else None,
                        body_type=body_type if body_type else None,
                        embedding=embedding,
                        image_path=uploaded_file.name
                    )
                    upload_time = time.time() - start_time
                
                progress_bar.progress(100)
                status_text.text("✓ Завършено!")
                
                # Success message
                st.success("✅ Превозно средство качено успешно!")
                
                # Results
                st.markdown("### 📋 Upload Details")
                
                col_x, col_y = st.columns(2)
                
                with col_x:
                    st.metric("Vehicle UUID", vehicle_uuid)
                    st.metric("Database ID", vehicle_id)
                    st.metric("Mode", "🔐 Encrypted" if is_encrypted else "🔓 Plain")
                
                with col_y:
                    st.metric("Embedding Time", f"{embedding_time:.3f}s")
                    st.metric("Upload Time", f"{upload_time:.3f}s")
                    st.metric("Total Time", f"{embedding_time + upload_time:.3f}s")
                
                # Embedding info
                with st.expander("🔢 Embedding Information"):
                    st.write(f"**Shape:** {embedding.shape}")
                    st.write(f"**Norm:** {np.linalg.norm(embedding):.6f}")
                    st.write(f"**First 10 values:**")
                    st.code(embedding[:10])
                    
                    if is_encrypted:
                        st.info("🔐 Embedding е криптиран с CKKS и normalized за PIR търсене")
                
            except Exception as e:
                st.error(f"❌ Грешка при качване: {e}")
                st.exception(e)

st.markdown("---")

# Tips
st.markdown("### 💡 Tips")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Plain Mode:**
    - ✅ Много бързо качване (~0.1s)
    - ✅ Бързо търсене (~10ms)
    - ❌ Сървърът вижда embedding-a
    - ❌ Не е private
    
    **Използвай за:** Development, testing
    """)

with col2:
    st.markdown("""
    **Encrypted Mode:**
    - ✅ PIR-ready криптиране
    - ✅ Query privacy при търсене
    - ❌ По-бавно качване (~1-2s)
    - ❌ По-бавно търсене (~500ms)
    
    **Използвай за:** Production, demo, privacy
    """)

# Statistics
st.markdown("---")
st.markdown("### 📊 Upload Statistics")

from database.operations import get_database_stats

try:
    stats = get_database_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Uploads", stats['total_vehicles'])
    col2.metric("Plain", stats['plain_count'])
    col3.metric("Encrypted", stats['encrypted_count'])
    col4.metric("Today", "0")  # Можеш да добавиш логика за това
    
except:
    st.warning("Не може да се заредят статистики")