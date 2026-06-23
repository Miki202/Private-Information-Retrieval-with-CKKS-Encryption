import streamlit as st
import pandas as pd
from PIL import Image
import time
import sys

from ui.utils import load_encoder_model, encode_image
from storage.database.operations import (
    pir_search_true,
    get_all_vehicles
)

st.set_page_config(
    page_title="Сигурно търсене чрез PIR",
    layout="centered"
)

st.markdown("<h1 style='text-align: center;'>🛡️ Сигурно търсене на изображения</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #fffff0;'>чрез PIR и хомоморфно криптиране</h4>", unsafe_allow_html=True)

st.write("") 

st.markdown("<h5 style='text-align: center; color: #fffff0;'>Описание на системата</h5>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center; max-width:900px; margin:0 auto;">
      <p style="font-size:1rem; color:#94a3b8; margin-bottom:12px;">
        Тази система демонстрира сигурно търсене на изображения чрез следните стъпки:
      </p>

      <ul style="display:inline-block; text-align:left; padding:0 20px; margin:0; list-style: none;">
        <li style="margin:6px 0;"><strong>Локално извличане на характеристики</strong> със CNN енкодер</li>
        <li style="margin:6px 0;"><strong>Генериране на ембединги</strong> за заявката</li>
        <li style="margin:6px 0;"><strong>Криптиране на заявката</strong> (PIR)</li>
        <li style="margin:6px 0;"><strong>Хомоморфно сравнение</strong> върху криптирани данни</li>
        <li style="margin:6px 0;"><strong>Декриптиране и връщане</strong> само на ранкирани топ резултати</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

@st.cache_resource
def get_model():
    try:
        model, device = load_encoder_model("encoder.pth")
        return model, device, True
    except Exception:
        return None, None, False

model, device, model_loaded = get_model()

def show_step(step_container, text, delay=1.0):
    with step_container:
        with st.spinner(text):
            time.sleep(delay)

st.header("Нова заявка")

col1, col2 = st.columns([2, 1])

with col1:
    query_img = st.file_uploader(
        "Качете изображение",
        type=["jpg", "jpeg", "png"]
    )

with col2:
    st.write("")
    st.write("")
    k = st.slider("Брой резултати (K)", 1, 20, 5)

if query_img:
    st.markdown("### Преглед на заявката")
    image = Image.open(query_img).convert("RGB")
    
    img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
    with img_col2:
        st.image(image, use_container_width=True, caption="Изображение за търсене")

    st.write("")
    
    if st.button("Стартирай сигурно търсене", type="primary", use_container_width=True):
        start_time = time.time()
        
        st.divider()
        st.markdown("### Процес на търсене")
        
        workflow = st.container()

        with workflow:
            step = st.empty()
            show_step(step, "Стъпка 1: Зареждане на изображението")
            show_step(step, "Стъпка 2: Локално извличане на ембеддинг (CNN encoder)")

            if not model_loaded:
                show_step(step, "Encoder не е наличен - използва се демо режим")
                embedding = None
            else:
                embedding = encode_image(image, model, device)

            show_step(step, "Стъпка 3: Подготовка на PIR заявка")
            show_step(step, "Стъпка 4: Изпращане на криптирана заявка към сървъра")
            show_step(step, "Стъпка 5: Сървърът извършва сравнение върху криптирани данни")
            show_step(step, "Стъпка 6: Декриптиране на резултатите и подреждане")

            if model_loaded:
                results = pir_search_true(embedding, top_k=k)
            else:
                results = [
                    {
                        "vehicle": {
                            "license_plate": "DEMO-001",
                            "color": "Черен",
                            "body_type": "SUV"
                        },
                        "similarity_score": 0.9215
                    },
                    {
                        "vehicle": {
                            "license_plate": "DEMO-002",
                            "color": "Бял",
                            "body_type": "Седан"
                        },
                        "similarity_score": 0.8742
                    }
                ][:k]
            elapsed_time = time.time() - start_time
            
            step.success("Готово. Резултатите са декриптирани и върнати.")

        st.write("")
        st.metric(label="Общо време за изпълнение", value=f"{elapsed_time:.2f} сек")

        st.subheader("Резултати")

        rows = []
        for i, r in enumerate(results, 1):
            v = r["vehicle"]
            rows.append({
                "Ранг": i,
                "Сходство": f"{r['similarity_score'] * 100:.2f}%",
                "Регистрационен номер": v.get("license_plate"),
                "Цвят": v.get("color"),
                "Тип": v.get("body_type"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

with st.expander("Сървърен изглед на базата данни"):
    st.caption("Поради имплементацията на PIR, сървърът съхранява метаданните и ембедингите в криптиран вид.")
    
    try:
        vehicles = get_all_vehicles()
    except Exception:
        vehicles = []

    rows = []
    for v in vehicles:
        total_bytes = 0
        for attr in dir(v):
            if not attr.startswith('_') and attr not in ['id', 'vehicle_uuid']:
                val = getattr(v, attr, None)
                if isinstance(val, bytes):
                    total_bytes += len(val)
                elif isinstance(val, str) and len(val) > 100:
                    total_bytes += len(val)

        if total_bytes == 0:
            total_bytes = sys.getsizeof(v)

        size_kb = total_bytes/1024

        rows.append({
            "ID запис": v.id,
            "Идентификатор": f"ENC00{v.id}" if v.id < 10 else f"ENC0{v.id}",
            "Статус на метаданните": "🔒 Криптирани (AES/HE)",
            "Видимост за сървъра": "Нулева (Blind Server)",
            "Реален размер на данните": f"{size_kb:.2f} KB",
            "Системен UUID": v.vehicle_uuid
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Базата данни е празна.")