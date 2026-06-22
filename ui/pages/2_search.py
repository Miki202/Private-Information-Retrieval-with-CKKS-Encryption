import streamlit as st
import os
from PIL import Image
import pandas as pd
import time
from storage.database.operations import pir_search_true, get_database_stats
from ui.pages.utils import load_encoder_model, encode_image

st.set_page_config(page_title="PIR Търсене", page_icon="🔍", layout="wide")

st.title("🔍 Поверително PIR Търсене")
st.caption("Заявката се шифрова преди изпращане. Сървърът извършва изчисления над шифъртекст и не знае какво търсите.")

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

stats = get_database_stats()
if stats["total_vehicles"] == 0:
    st.warning("Базата данни е празна. Моля, първо качете автомобили.")
    st.stop()

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    with st.container(border=True):
        st.markdown("### 🕵️‍♂️ Входна заявка")
        query_file = st.file_uploader("Качете изображение за търсене", type=["jpg", "jpeg", "png"])
        if query_file:
            image = Image.open(query_file).convert("RGB")
            st.image(image, caption="Заявка", use_container_width=True)

    with st.container(border=True):
        st.markdown("### ⚙️ Параметри")
        top_k = st.slider("Брой резултати за връщане", 1, 50, 10)
        
        st.markdown("**Филтри след декриптиране** *(прилагат се локално)*")
        filter_color = st.selectbox("Цвят", ["Всички", "червен", "син", "зелен", "черен", "бял", "сребрист", "сив", "жълт", "друг"])
        filter_body = st.selectbox("Тип купе", ["Всички", "седан", "SUV / Джип", "камион", "ван", "купе", "хечбек", "комби", "друг"])

        search_button = st.button("🔐 Стартирай PIR Търсене", type="primary", disabled=query_file is None, use_container_width=True)

with col2:
    st.markdown("### 📊 Резултати от търсенето")
    
    if search_button and query_file:
        with st.spinner("Генериране на векторен отпечатък и шифроване..."):
            t0 = time.time()
            embedding = encode_image(image, model, device)
            embed_time = time.time() - t0

        fetch_k = top_k * 5 if (filter_color != "Всички" or filter_body != "Всички") else top_k

        with st.spinner("Извършване на хомоморфно сканиране на сървъра..."):
            t0 = time.time()
            results = pir_search_true(embedding, top_k=fetch_k, verbose=False)
            search_time = time.time() - t0

        if filter_color != "Всички":
            results = [r for r in results if r["vehicle"].get("color") == filter_color]
        if filter_body != "Всички":
            results = [r for r in results if r["vehicle"].get("body_type") == filter_body]
        results = results[:top_k]

        m1, m2, m3 = st.columns(3)
        m1.metric("Векторизация", f"{embed_time:.3f} с")
        m2.metric("PIR Сканиране", f"{search_time:.3f} с")
        m3.metric("Намерени", len(results))

        if not results:
            st.warning("Няма намерени резултати с избраните филтри.")
        else:
            for i, r in enumerate(results):
                v = r["vehicle"]
                score = r["similarity_score"]
                
                with st.expandable_container(f"#{i+1} [{v.get('license_plate') or 'БЕЗ НОМЕР'}] — Сходство: {score:.4f}", expanded=i < 2):
                    col_l, col_r = st.columns([1, 2])
                    with col_l:
                        st.metric("Точно съвпадение", f"{score:.5f}")
                    with col_r:
                        st.markdown(f"**Рег. номер:** `{v.get('license_plate') or 'N/A'}`")
                        st.markdown(f"**Цвят:** {v.get('color') or 'N/A'} | **Купе:** {v.get('body_type') or 'N/A'}")
                        st.caption(f"UUID: {v.get('uuid')}")

            df = pd.DataFrame([{
                "Ранг": i + 1,
                "Рег. номер": r["vehicle"].get("license_plate"),
                "Цвят": r["vehicle"].get("color"),
                "Купе": r["vehicle"].get("body_type"),
                "Сходство": f"{r['similarity_score']:.6f}"
            } for i, r in enumerate(results)])
            
            st.download_button("📥 Изтегли резултатите (CSV)", df.to_csv(index=False), "pir_results.csv", "text/csv", use_container_width=True)
    else:
        st.info("Моля, качете изображение отляво и натиснете бутона за търсене.")