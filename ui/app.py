import streamlit as st
import os
import time
import pandas as pd
import numpy as np
from PIL import Image

from storage.database.operations import (
    insert_vehicle,
    pir_search_true,
    get_database_stats,
    get_encryption_stats,
    get_all_vehicles,
    delete_vehicle
)
from storage.database.encryption import decrypt_metadata
from ui.pages.utils import load_encoder_model, encode_image

st.set_page_config(
    page_title="PIR Система - Академичен Прототип",
    layout="wide",
    initial_sidebar_state="expanded" 
)

with st.sidebar:
    st.markdown("### ⚙️ Системен статус")
    st.success("🟢 Активна защитена сесия")
    
    st.markdown("---")
    
    st.markdown("### 📊 Технически параметри")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.metric(label="Вектор", value="256-D")
    with sc2:
        st.metric(label="Мрежа", value="Autoencoder")
        
    st.metric(label="Хомоморфна схема", value="CKKS")
    st.metric(label="Шифър на метаданни", value="AES-256")

    st.markdown("---")
    
    st.markdown("### 📚 Кратка справка за използваните технологии в прототипа:")
    
    with st.expander("🔍 Какво е PIR?"):
        st.markdown("""
        **Private Information Retrieval (PIR)** е криптографски протокол, който позволява 
        на потребителя да извлече информация от база данни, без сървърът да разбере 
        конкретно какво е било потърсено или извлечено.
        """)
        
    with st.expander("🔐 Какво е CKKS?"):
        st.markdown("""
        **CKKS** (Cheon-Kim-Kim-Song) е схема за **хомоморфно шифроване**. 
        Тя позволява извършването на математически операции (като събиране и умножение) 
        директно върху криптирани числа (шифъртекст), без те да бъдат дешифрирани.
        """)
        

    st.markdown("---")
    st.caption("Разработено за изследователски цели. Ключовете за дешифриране не напускат това устройство.")

st.markdown("<h1 style='text-align: center;'>Система за поверително извличане на информация (PIR) чрез хомоморфно шифроване</h1>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
**Абстракт:** Настоящият софтуерен прототип демонстрира архитектура за поверително търсене на обекти (автомобили) в отдалечена база данни. 
Системата използва изкуствен интелект за извличане на векторни характеристики и **CKKS хомоморфно шифроване**, което позволява 
изчисляване на скаларно произведение директно върху шифъртекст. По този начин сървърът изпълнява заявката без да дешифрира 
входните данни, гарантирайки пълна криптографска анонимност на потребителя.

**Методология:**
1. **Въвеждане на данни:** Изображенията се векторизират (256-размерен вектор) и се криптират локално преди запис в базата.
2. **Поверително търсене:** Заявката се шифрова при клиента. Сървърът изчислява сходството спрямо всички записи (скривайки модела на достъп).
3. **Дешифриране:** Резултатите се връщат като шифъртекст и се дешифрират локално за визуализация.
""")
st.markdown("---")

@st.cache_resource
def get_model():
    model_path = "encoder.pth"
    if not os.path.exists(model_path):
        st.error(f"Грешка: Моделът не е намерен на адрес: {model_path}")
        return None, None
    try:
        return load_encoder_model(model_path)
    except Exception as e:
        st.error(f"Системна грешка при инициализация на невронната мрежа: {e}")
        return None, None

model, device = get_model()
if model is None:
    st.stop()

tab_upload, tab_search, tab_browse = st.tabs([
    "Раздел I: Въвеждане на данни", 
    "Раздел II: Поверително търсене", 
    "Раздел III: Преглед и статистика"
])

with tab_upload:
    st.markdown("### Инициализация на нови записи в базата данни")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("**Входно изображение**")
        uploaded_file = st.file_uploader("Избор на файл", type=["jpg", "jpeg", "png"], key="upload_img")
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Визуализация на входните данни", use_container_width=True)

    with col2:
        st.markdown("**Метаданни (Локално шифроване)**")
        license_plate = st.text_input("Регистрационен номер", placeholder="СА1234ХХ").upper()
        
        c_a, c_b = st.columns(2)
        with c_a:
            color = st.selectbox("Цвят", ["", "червен", "син", "зелен", "черен", "бял", "сребрист", "сив", "жълт", "друг"])
        with c_b:
            body_type = st.selectbox("Тип купе", ["", "седан", "SUV / Джип", "камион", "ван", "купе", "хечбек", "комби", "друг"])

    if uploaded_file:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Шифроване и качване в сървъра", type="primary", use_container_width=True):
            with st.status("Изпълнение на криптографски протокол...", expanded=True) as status:
                try:
                    status.write("Стъпка 1: Извличане на векторен отпечатък...")
                    t0 = time.time()
                    embedding = encode_image(image, model, device)
                    embed_time = time.time() - t0

                    status.write("Стъпка 2: CKKS хомоморфно шифроване и трансфер...")
                    t0 = time.time()
                    vehicle_uuid, vehicle_id = insert_vehicle(
                        embedding=embedding,
                        license_plate=license_plate or None,
                        color=color or None,
                        body_type=body_type or None,
                        image_path=uploaded_file.name,
                    )
                    upload_time = time.time() - t0
                    
                    status.update(label="Процесът завърши успешно.", state="complete", expanded=False)
                    st.success(f"Записът е съхранен. Време за векторизация: {embed_time:.3f}s | Време за криптиране: {upload_time:.3f}s")
                except Exception as e:
                    status.update(label="Възникна грешка в протокола.", state="error")
                    st.error(f"Системна грешка: {e}")

with tab_search:
    st.markdown("### Извличане на информация чрез изчисляване върху шифъртекст")
    
    col_s1, col_s2 = st.columns([1, 2], gap="large")

    with col_s1:
        st.markdown("**Заявка (Търсено изображение)**")
        query_file = st.file_uploader("Избор на файл за заявка", type=["jpg", "jpeg", "png"], key="search_img")
        if query_file:
            query_image = Image.open(query_file).convert("RGB")
            st.image(query_image, caption="Обект на търсене", use_container_width=True)

        st.markdown("**Параметри на извличането**")
        top_k = st.slider("Брой целеви резултати (k)", 1, 50, 10)
        
        st.markdown("*Локални филтри (прилагат се след дешифриране):*")
        filter_color = st.selectbox("Изискван цвят", ["Всички", "червен", "син", "зелен", "черен", "бял", "сребрист", "сив", "жълт", "друг"])
        filter_body = st.selectbox("Изискван тип купе", ["Всички", "седан", "SUV / Джип", "камион", "ван", "купе", "хечбек", "комби", "друг"])

        search_button = st.button("Иницииране на PIR заявка", type="primary", disabled=query_file is None, use_container_width=True)

    with col_s2:
        st.markdown("**Резултати от хомоморфното сканиране**")
        
        if search_button and query_file:
            with st.spinner("Генериране и шифроване на заявката..."):
                t0 = time.time()
                query_embedding = encode_image(query_image, model, device)
                embed_time = time.time() - t0

            fetch_k = top_k * 5 if (filter_color != "Всички" or filter_body != "Всички") else top_k

            with st.spinner("Сървърно изчисляване (Скаларно произведение на шифъртекст)..."):
                t0 = time.time()
                results = pir_search_true(query_embedding, top_k=fetch_k, verbose=False)
                search_time = time.time() - t0

            if filter_color != "Всички":
                results = [r for r in results if r["vehicle"].get("color") == filter_color]
            if filter_body != "Всички":
                results = [r for r in results if r["vehicle"].get("body_type") == filter_body]
            results = results[:top_k]

            st.info(f"Времеви метрики: Векторизация ({embed_time:.3f}s) | Хомоморфно сканиране ({search_time:.3f}s)")

            if not results:
                st.warning("Не са открити записи, отговарящи на зададените критерии.")
            else:
                for i, r in enumerate(results):
                    v = r["vehicle"]
                    score = r["similarity_score"]
                    
                    with st.expander(f"Ранг {i+1} | Сходство: {score:.5f} | Номер: {v.get('license_plate') or 'N/A'}", expanded=i < 2):
                        st.markdown(f"""
                        - **Регистрационен номер:** {v.get('license_plate') or 'Неизвестен'}
                        - **Физически характеристики:** Цвят: {v.get('color') or 'N/A'}, Купе: {v.get('body_type') or 'N/A'}
                        - **Системен идентификатор:** `{v.get('uuid')}`
                        """)
                
                df_results = pd.DataFrame([{
                    "Ранг": i + 1,
                    "Идентификатор": r["vehicle"].get("uuid"),
                    "Рег. номер": r["vehicle"].get("license_plate"),
                    "Сходство": f"{r['similarity_score']:.6f}"
                } for i, r in enumerate(results)])
                
                st.download_button("Експорт на резултатите (CSV)", df_results.to_csv(index=False), "pir_results.csv", "text/csv")
        elif not query_file:
            st.caption("Очаква се въвеждане на изображение за иницииране на заявката.")

with tab_browse:
    st.markdown("### Анализ на съхранената криптографска база данни")
    
    try:
        general_stats = get_database_stats()
        enc_stats = get_encryption_stats()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Общ брой векторни записи", general_stats["total_vehicles"])
        m2.metric("Среден обем на шифъртекст", f"{general_stats['avg_encrypted_storage_bytes']:,} Байта")
        m3.metric("Криптографски разход (Overhead)", f"{enc_stats['storage_overhead']:.1f}x спрямо прав текст")
    except Exception:
        st.warning("Статистическите данни не са налични в момента.")

    st.markdown("---")
    
    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        page_size = st.selectbox("Размер на извадката", [10, 25, 50, 100])
    with col_p2:
        page = st.number_input("Текуща страница", min_value=1, value=1, step=1)

    try:
        vehicles = get_all_vehicles(skip=(page - 1) * page_size, limit=page_size)
        if not vehicles:
            st.info("Базата данни е празна.")
        else:
            rows = []
            for v in vehicles:
                try:
                    meta = decrypt_metadata(v.encrypted_metadata)
                except Exception:
                    meta = {}
                rows.append({
                    "Системно ID": v.id,
                    "UUID (Кратък)": str(v.vehicle_uuid)[:8] + "...",
                    "Рег. номер": meta.get("license_plate") or "N/A",
                    "Цвят": meta.get("color") or "N/A",
                    "Тип купе": meta.get("body_type") or "N/A",
                    "Добавен на": v.created_at.strftime("%Y-%m-%d %H:%M") if v.created_at else "N/A",
                    "_full_uuid": v.vehicle_uuid
                })

            df = pd.DataFrame(rows)
            st.dataframe(df.drop(columns=["_full_uuid"]), use_container_width=True, hide_index=True)

            st.markdown("#### Административни действия")
            selected_id = st.selectbox("Избор на запис за премахване (по UUID)", df["_full_uuid"])
            confirm = st.checkbox("Потвърждавам изтриването на този запис (необратимо действие)")
            if st.button("Премахване от базата", type="primary", disabled=not confirm):
                if delete_vehicle(selected_id):
                    st.success("Записът е заличен успешно.")
                    st.rerun()

    except Exception as e:
        st.error(f"Грешка при извличане на данните: {e}")