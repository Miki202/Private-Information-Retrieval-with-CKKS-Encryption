import streamlit as st
import pandas as pd
from storage.database.operations import get_all_vehicles, delete_vehicle, get_database_stats, get_encryption_stats
from storage.database.encryption import decrypt_metadata

st.set_page_config(page_title="Преглед на базата", page_icon="📊", layout="wide")

st.title("📊 Управление и Преглед на Автомобили")
st.caption("Метаданните се декриптират локално във вашия браузър. Базата данни съхранява само криптографски текст.")

tab_list, tab_details = st.tabs(["📋 Списък с автомобили", "⚙️ Статистика на криптирането"])

try:
    general_stats = get_database_stats()
    enc_stats = get_encryption_stats()
except Exception:
    general_stats, enc_stats = None, None

with tab_list:
    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        page_size = st.selectbox("Елементи на страница", [10, 25, 50, 100], index=0)
    with col_p2:
        page = st.number_input("Страница", min_value=1, value=1, step=1)

    skip = (page - 1) * page_size

    try:
        vehicles = get_all_vehicles(skip=skip, limit=page_size)
        if not vehicles:
            st.warning("Няма намерени записи на тази страница.")
        else:
            rows = []
            for v in vehicles:
                try:
                    meta = decrypt_metadata(v.encrypted_metadata)
                except Exception:
                    meta = {}
                rows.append({
                    "_uuid": v.vehicle_uuid,
                    "ID": v.id,
                    "Рег. номер": meta.get("license_plate") or "N/A",
                    "Цвят": meta.get("color") or "N/A",
                    "Тип купе": meta.get("body_type") or "N/A",
                    "Файл": meta.get("image_path") or "N/A",
                    "Дата на създаване": v.created_at.strftime("%Y-%m-%d %H:%M") if v.created_at else "N/A"
                })

            df = pd.DataFrame(rows)
            st.dataframe(df.drop(columns=["_uuid"]), use_container_width=True, hide_index=True)

            st.markdown("### 🛠️ Действия за избран запис")
            selected_plate = st.selectbox("Изберете автомобил за управление", df["Рег. номер"].unique())
            selected_row = df[df["Рег. number"] == selected_plate].iloc[0] if not df.empty else None

            if selected_row is not None:
                c1, c2 = st.columns(2)
                with c1:
                    st.json({
                        "Идентификатор (UUID)": str(selected_row["_uuid"]),
                        "Дата": selected_row["Дата на създаване"],
                        "Име на файл": selected_row["Файл"]
                    })
                with c2:
                    st.error("⚠️ Зона за сигурност: Изтриването е необратимо.")
                    confirm = st.checkbox(f"Потвърждавам изтриването на {selected_plate}")
                    if st.button("🗑️ Изтрий записа", type="primary", disabled=not confirm, use_container_width=True):
                        if delete_vehicle(selected_row["_uuid"]):
                            st.success("Записът е изтрит успешно.")
                            st.rerun()

    except Exception as e:
        st.error(f"Грешка при зареждане на данните: {e}")

with tab_details:
    if general_stats and enc_stats:
        m1, m2, m3 = st.columns(3)
        m1.metric("Общ брой обекти", general_stats["total_vehicles"])
        m2.metric("Размер на криптиран запис", f"{general_stats['avg_encrypted_storage_bytes']:,} B")
        m3.metric("Криптографско забавяне/нарастване", f"{enc_stats['storage_overhead']:.1f}x")