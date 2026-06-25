import streamlit as st
import pandas as pd
from PIL import Image
import time
import sys

from utils import encode_image
from storage.database.operations import (
    pir_search_true,
    get_all_vehicles
)
from storage.database.connection import get_db
from storage.database.models import Vehicle

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
        <li style="margin:6px 0;"><strong>Stage 1: Bucketization</strong> — филтриране по цвят / рег. номер</li>
        <li style="margin:6px 0;"><strong>Stage 2: Криптиране на заявката</strong> (PIR)</li>
        <li style="margin:6px 0;"><strong>Хомоморфно сравнение</strong> върху криптирани данни</li>
        <li style="margin:6px 0;"><strong>Декриптиране и връщане</strong> само на ранкирани топ резултати</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

model_loaded = True


def show_step(step_container, text, delay=1.0):
    with step_container:
        with st.spinner(text):
            time.sleep(delay)

@st.cache_data(ttl=60)
def get_available_colors():
    """Зарежда уникалните стойности за цвят от базата."""
    db = get_db()
    try:
        colors = db.query(Vehicle.color).distinct().filter(Vehicle.color.isnot(None)).all()
        return sorted([c[0] for c in colors if c[0]])
    finally:
        db.close()


try:
    available_colors = get_available_colors()
except Exception:
    available_colors = []

st.header("Нова заявка")

col1, col2 = st.columns([2, 1])

with col1:
    query_img = st.file_uploader(
        "Качете изображение",
        type=["jpg", "jpeg", "png"]
    )

with col2:
    st.write("")
    k = st.slider("Брой резултати (K)", 1, 20, 5)

    st.markdown("##### Stage 1: Bucketization")
    filter_color = st.selectbox(
        "Филтър по цвят",
        ["Всички"] + available_colors
    )
    filter_plate = st.text_input(
        "Филтър по рег. номер",
        placeholder="напр. AA1234BB"
    )

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
            show_step(step, "Стъпка 1: Зареждане на изображението", delay=0.5)
            show_step(step, "Стъпка 2: Локално извличане на ембеддинг (CNN encoder)", delay=0.5)

            if not model_loaded:
                show_step(step, "Encoder не е наличен — използва се демо режим")
                embedding = None
            else:
                embedding = encode_image(image)

            # Stage 1 филтри
            color_to_filter = filter_color if filter_color != "Всички" else None
            plate_to_filter = filter_plate.strip() if filter_plate.strip() else None

            if color_to_filter or plate_to_filter:
                filters_text = []
                if color_to_filter:
                    filters_text.append(f"цвят='{color_to_filter}'")
                if plate_to_filter:
                    filters_text.append(f"номер='{plate_to_filter}'")
                show_step(step, f"Стъпка 3: Stage 1 — Bucketization ({', '.join(filters_text)})", delay=0.5)
            else:
                show_step(step, "Стъпка 3: Stage 1 — Без филтри (пълно сканиране)", delay=0.5)

            show_step(step, "Стъпка 4: Криптиране на заявката (CKKS)", delay=0.5)
            show_step(step, "Стъпка 5: Stage 2 — Хомоморфно сравнение върху криптирани данни", delay=0.5)

            if model_loaded and embedding is not None:
                results = pir_search_true(
                    embedding,
                    top_k=k,
                    filter_color=color_to_filter,
                    filter_plate=plate_to_filter,
                    verbose=True,
                )
            else:
                results = []

            show_step(step, "Стъпка 6: Декриптиране на резултатите и подреждане", delay=0.5)

            elapsed_time = time.time() - start_time

            step.success("Готово. Резултатите са декриптирани и върнати.")

        st.write("")

        # Метрики
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Време", f"{elapsed_time:.2f} сек")
        col_m2.metric("Резултати", len(results))
        col_m3.metric("Bucket филтри", f"{int(bool(color_to_filter)) + int(bool(plate_to_filter))}/2")

        # Резултати
        st.subheader("Резултати")

        if results:
            rows = []
            for i, r in enumerate(results, 1):
                v = r["vehicle"]
                rows.append({
                    "#": i,
                    "UUID": v.get("uuid", "—"),
                    "Рег. номер": v.get("license_plate") or "—",
                    "Цвят": v.get("color") or "—",
                    "Сходство": f"{r['similarity_score']:.4f}"
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("Няма намерени резултати.")

st.divider()
with st.expander("Сървърен изглед на базата данни"):
    st.caption("Поради имплементацията на PIR, сървърът съхранява метаданните и ембедингите в криптиран вид.")

    try:
        vehicles = get_all_vehicles()
    except Exception:
        vehicles = []

    rows = []
    for v in vehicles:
        size_kb = v.get_storage_size() / 1024

        rows.append({
            "ID": v.id,
            "Рег. номер (криптиран)": f"🔒 {len(v.encrypted_metadata)} bytes",
            "Ембединг (криптиран)": f"🔒 {len(v.encrypted_embedding)} bytes",
            "Bucket: Цвят": v.color or "—",
            "Bucket: Номер": v.license_plate or "—",
            "Криптиран размер": f"{size_kb:.1f} KB",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.info(f"Общо записи: {len(rows)} | Сървърът вижда САМО bucket колоните и криптираните blob-ове.")
    else:
        st.info("Базата данни е празна.")
