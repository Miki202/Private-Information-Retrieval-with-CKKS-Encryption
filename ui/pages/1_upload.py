import streamlit as st
import os
from PIL import Image
import numpy as np
import time
from storage.database.operations import insert_vehicle, get_database_stats
from ui.pages.utils import load_encoder_model, encode_image

st.set_page_config(page_title="Качване на автомобил", page_icon="📤", layout="wide")

st.title("📤 Качване на превозно средство")
st.caption("Всички данни се шифроват клиентски чрез CKKS хомоморфно шифроване преди запис.")

@st.cache_resource
def get_model():
    model_path = "encoder.pth"
    if not os.path.exists(model_path):
        st.error(f"Моделът не е намерен на адрес: {model_path}")
        return None, None
    try:
        return load_encoder_model(model_path)
    except Exception as e:
        st.error(f"Грешка при зареждане на модела: {e}")
        return None, None

model, device = get_model()
if model is None:
    st.stop()

with st.sidebar:
    st.success("Невронната мрежа е заредена успешно.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    with st.container(border=True):
        st.markdown("### 📸 Изображение")
        uploaded_file = st.file_uploader(
            "Изберете снимка на автомобила",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_container_width=True)
            st.caption(f"Разделителна способност: {image.size[0]} × {image.size[1]} px")

with col2:
    with st.container(border=True):
        st.markdown("### 📝 Метаданни")
        license_plate = st.text_input("Регистрационен номер", placeholder="СА1234ХХ").upper()
        
        col_a, col_b = st.columns(2)
        with col_a:
            color = st.selectbox(
                "Цвят",
                ["", "червен", "син", "зелен", "черен", "бял", "сребрист", "сив", "жълт", "друг"],
            )
        with col_b:
            body_type = st.selectbox(
                "Тип купе",
                ["", "седан", "SUV / Джип", "камион", "ван", "купе", "хечбек", "комби", "друг"],
            )

if uploaded_file:
    upload_button = st.button(
        "🚀 Шифроване и Качване",
        type="primary",
        use_container_width=True,
    )

    if upload_button:
        with st.status("Обработка на данните...", expanded=True) as status:
            try:
                status.write("Генериране на векторен еквивалент (Embedding)...")
                t0 = time.time()
                embedding = encode_image(image, model, device)
                embed_time = time.time() - t0

                status.write("Хомоморфно шифроване и запис в базата данни...")
                t0 = time.time()
                vehicle_uuid, vehicle_id = insert_vehicle(
                    embedding=embedding,
                    license_plate=license_plate or None,
                    color=color or None,
                    body_type=body_type or None,
                    image_path=uploaded_file.name,
                )
                upload_time = time.time() - t0
                
                status.update(label="Успешно качване!", state="complete", expanded=False)
                st.success("Данните са шифровани успешно на вашето устройство и са изпратени!")

                m1, m2, m3 = st.columns(3)
                m1.metric("База данни ID", vehicle_id)
                m2.metric("Векторизация", f"{embed_time:.3f} сек")
                m3.metric("Криптиране + Запис", f"{upload_time:.3f} сек")

                with st.expander("Детайли за шифрования вектор"):
                    st.write(f"Размерност: {embedding.shape}")
                    st.write(f"Норма: {np.linalg.norm(embedding):.6f}")
                    st.code(str(embedding[:5]) + " ...")
                    st.info("Сървърът вижда само шифрования шифъртекст (Ciphertext). Векторът в прав текст не напуска браузъра.")

            except Exception as e:
                status.update(label="Грешка при качването!", state="error")
                st.error(f"Неуспешно качване: {e}")