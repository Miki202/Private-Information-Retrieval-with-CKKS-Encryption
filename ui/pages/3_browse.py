"""
Browse страница - преглед на всички превозни средства
"""
import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from database.operations import (
    get_all_vehicles,
    delete_vehicle,
    get_database_stats,
    get_encryption_stats,
    convert_to_encrypted
)

st.set_page_config(page_title="Browse Vehicles", page_icon="📊", layout="wide")

st.title("📊 Browse All Vehicles")
st.markdown("Преглед и управление на всички превозни средства в базата данни")

st.markdown("---")

# Statistics
st.markdown("### 📈 Database Overview")

try:
    general_stats = get_database_stats()
    encryption_stats = get_encryption_stats()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("🚗 Total", general_stats['total_vehicles'])
    col2.metric("🔓 Plain", general_stats['plain_count'])
    col3.metric("🔐 Encrypted", general_stats['encrypted_count'])
    col4.metric("📊 Vectors", general_stats['total_vectors'])
    col5.metric("🔐 Rate", f"{encryption_stats['encryption_percentage']:.0f}%")
    
except Exception as e:
    st.error(f"❌ Грешка при статистика: {e}")

st.markdown("---")

# Filters and Pagination
st.markdown("### 🔧 Filters & Settings")

col1, col2, col3, col4 = st.columns(4)

with col1:
    page_size = st.selectbox("Items per page", [10, 25, 50, 100], index=1)

with col2:
    filter_mode = st.selectbox("Mode", ["All", "Plain Only", "Encrypted Only"])

with col3:
    filter_color_browse = st.selectbox(
        "Color",
        ["All"] + list(general_stats.get('color_distribution', {}).keys())
    )

with col4:
    filter_body_browse = st.selectbox(
        "Body Type",
        ["All"] + list(general_stats.get('body_type_distribution', {}).keys())
    )

# Page number
page = st.number_input("Page", min_value=1, value=1, step=1)
skip = (page - 1) * page_size

st.markdown("---")

# Load vehicles
st.markdown("### 🚗 Vehicle List")

try:
    vehicles = get_all_vehicles(skip=skip, limit=page_size)
    
    if not vehicles:
        st.warning("📭 Няма превозни средства в базата данни.")
        st.info("💡 Отидете на Upload страницата за да добавите превозни средства.")
    else:
        # Apply filters
        filtered_vehicles = vehicles
        
        # Mode filter
        if filter_mode == "Plain Only":
            filtered_vehicles = [v for v in filtered_vehicles if not v.is_encrypted]
        elif filter_mode == "Encrypted Only":
            filtered_vehicles = [v for v in filtered_vehicles if v.is_encrypted]
        
        # Color filter
        if filter_color_browse != "All":
            filtered_vehicles = [v for v in filtered_vehicles if v.color == filter_color_browse]
        
        # Body type filter
        if filter_body_browse != "All":
            filtered_vehicles = [v for v in filtered_vehicles if v.body_type == filter_body_browse]
        
        st.info(f"Показани {len(filtered_vehicles)} от {len(vehicles)} превозни средства на тази страница")
        
        # Convert to dataframe
        data = []
        for v in filtered_vehicles:
            data.append({
                "ID": v.id,
                "UUID": v.vehicle_uuid[:8] + "...",  # Съкратено
                "License Plate": v.license_plate or "N/A",
                "Color": v.color or "N/A",
                "Body Type": v.body_type or "N/A",
                "Mode": "🔐 Encrypted" if v.is_encrypted else "🔓 Plain",
                "Created": v.created_at.strftime("%Y-%m-%d %H:%M") if v.created_at else "N/A"
            })
        
        df = pd.DataFrame(data)
        
        # Display table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        col_dl1, col_dl2 = st.columns([1, 4])
        with col_dl1:
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "vehicles.csv",
                "text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Detailed view
        st.markdown("### 🔍 Detailed View")
        
        selected_uuid = st.selectbox(
            "Select vehicle to view details:",
            options=[v.vehicle_uuid for v in filtered_vehicles],
            format_func=lambda x: f"{next((v.license_plate for v in filtered_vehicles if v.vehicle_uuid == x), 'N/A')} - {x[:8]}..."
        )
        
        if selected_uuid:
            selected = next((v for v in filtered_vehicles if v.vehicle_uuid == selected_uuid), None)
            
            if selected:
                col_det1, col_det2 = st.columns([1, 1])
                
                with col_det1:
                    st.markdown("#### 📋 Vehicle Information")
                    st.write(f"**ID:** {selected.id}")
                    st.write(f"**UUID:** `{selected.vehicle_uuid}`")
                    st.write(f"**License Plate:** {selected.license_plate or 'N/A'}")
                    st.write(f"**Color:** {selected.color or 'N/A'}")
                    st.write(f"**Body Type:** {selected.body_type or 'N/A'}")
                    st.write(f"**Image Path:** {selected.image_path or 'N/A'}")
                    st.write(f"**Mode:** {'🔐 Encrypted' if selected.is_encrypted else '🔓 Plain'}")
                    st.write(f"**Created:** {selected.created_at}")
                    st.write(f"**Updated:** {selected.updated_at}")
                
                with col_det2:
                    st.markdown("#### ⚙️ Actions")
                    
                    # Convert to encrypted
                    if not selected.is_encrypted:
                        st.markdown("**Convert to Encrypted:**")
                        if st.button(f"🔐 Convert {selected.vehicle_uuid[:8]}... to Encrypted", key="convert"):
                            with st.spinner("Конвертиране..."):
                                try:
                                    success = convert_to_encrypted(selected.vehicle_uuid)
                                    if success:
                                        st.success("✅ Конвертирано успешно!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Конвертирането беше неуспешно")
                                except Exception as e:
                                    st.error(f"❌ Грешка: {e}")
                    else:
                        st.info("🔐 Това превозно средство вече е криптирано")
                    
                    st.markdown("---")
                    
                    # Delete
                    st.markdown("**Delete Vehicle:**")
                    st.warning("⚠️ Това действие е необратимо!")
                    
                    confirm_delete = st.checkbox(f"Потвърдете изтриването на {selected.license_plate or selected.vehicle_uuid[:8]}")
                    
                    if st.button(
                        f"🗑️ Delete {selected.vehicle_uuid[:8]}...",
                        type="secondary",
                        disabled=not confirm_delete,
                        key="delete"
                    ):
                        with st.spinner("Изтриване..."):
                            try:
                                success = delete_vehicle(selected.vehicle_uuid)
                                if success:
                                    st.success("✅ Превозното средство е изтрито!")
                                    st.rerun()
                                else:
                                    st.error("❌ Превозното средство не е намерено")
                            except Exception as e:
                                st.error(f"❌ Грешка: {e}")

except Exception as e:
    st.error(f"❌ Грешка при зареждане: {e}")
    st.exception(e)

st.markdown("---")

# Visualizations
st.markdown("### 📊 Data Visualizations")

try:
    stats = get_database_stats()
    
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        # Color distribution
        if stats['color_distribution']:
            st.markdown("#### Color Distribution")
            color_df = pd.DataFrame([
                {"Color": k, "Count": v}
                for k, v in stats['color_distribution'].items()
            ])
            fig_color = px.pie(
                color_df,
                values='Count',
                names='Color',
                title='Vehicles by Color'
            )
            st.plotly_chart(fig_color, use_container_width=True)
        else:
            st.info("Няма данни за цветове")
    
    with col_vis2:
        # Body type distribution
        if stats['body_type_distribution']:
            st.markdown("#### Body Type Distribution")
            body_df = pd.DataFrame([
                {"Body Type": k, "Count": v}
                for k, v in stats['body_type_distribution'].items()
            ])
            fig_body = px.bar(
                body_df,
                x='Body Type',
                y='Count',
                title='Vehicles by Body Type'
            )
            st.plotly_chart(fig_body, use_container_width=True)
        else:
            st.info("Няма данни за типове каросерия")
    
    # Encryption status
    st.markdown("#### Encryption Status")
    enc_df = pd.DataFrame([
        {"Status": "Plain", "Count": stats['plain_count']},
        {"Status": "Encrypted", "Count": stats['encrypted_count']}
    ])
    fig_enc = px.pie(
        enc_df,
        values='Count',
        names='Status',
        title='Plain vs Encrypted',
        color='Status',
        color_discrete_map={'Plain': '#ff7f0e', 'Encrypted': '#2ca02c'}
    )
    st.plotly_chart(fig_enc, use_container_width=True)

except Exception as e:
    st.error(f"❌ Грешка при визуализации: {e}")

st.markdown("---")

# Storage Statistics
st.markdown("### 💾 Storage Statistics")

try:
    enc_stats = get_encryption_stats()
    
    col_stor1, col_stor2, col_stor3 = st.columns(3)
    
    with col_stor1:
        st.metric(
            "Avg Encrypted Size",
            f"{enc_stats['avg_encrypted_size_bytes']:,} bytes",
            help="Среден размер на криптиран embedding"
        )
    
    with col_stor2:
        st.metric(
            "Avg Context Size",
            f"{enc_stats['avg_context_size_bytes']:,} bytes",
            help="Среден размер на CKKS context"
        )
    
    with col_stor3:
        overhead = enc_stats['avg_total_size_bytes'] / max(enc_stats['plain_embedding_size_bytes'], 1)
        st.metric(
            "Storage Overhead",
            f"{overhead:.1f}x",
            help="Колко пъти повече пространство заема encrypted спрямо plain"
        )
    
    # Detailed breakdown
    with st.expander("📊 Detailed Storage Breakdown"):
        st.write(f"**Plain embedding size:** {enc_stats['plain_embedding_size_bytes']:,} bytes (256 floats)")
        st.write(f"**Encrypted embedding size:** {enc_stats['avg_encrypted_size_bytes']:,} bytes")
        st.write(f"**Context size:** {enc_stats['avg_context_size_bytes']:,} bytes")
        st.write(f"**Total per vehicle:** {enc_stats['avg_total_size_bytes']:,} bytes")
        st.write(f"**Overhead:** {overhead:.2f}x")
        
        # Projections
        st.markdown("**Storage Projections:**")
        for num_vehicles in [1000, 10000, 100000, 1000000]:
            plain_total = num_vehicles * enc_stats['plain_embedding_size_bytes']
            encrypted_total = num_vehicles * enc_stats['avg_total_size_bytes']
            
            st.write(f"- **{num_vehicles:,} vehicles:**")
            st.write(f"  - Plain: {plain_total / 1024 / 1024:.1f} MB")
            st.write(f"  - Encrypted: {encrypted_total / 1024 / 1024:.1f} MB")

except Exception as e:
    st.error(f"❌ Грешка при storage статистика: {e}")

st.markdown("---")

# Bulk operations
st.markdown("### ⚙️ Bulk Operations")

col_bulk1, col_bulk2 = st.columns(2)

with col_bulk1:
    st.markdown("#### Convert All to Encrypted")
    st.write(f"Plain vehicles: {general_stats['plain_count']}")
    
    if st.button("🔐 Convert All Plain → Encrypted", type="secondary"):
        if general_stats['plain_count'] == 0:
            st.info("Няма plain превозни средства за конвертиране")
        else:
            with st.spinner(f"Конвертиране на {general_stats['plain_count']} превозни средства..."):
                try:
                    plain_vehicles = [v for v in get_all_vehicles(limit=1000) if not v.is_encrypted]
                    
                    progress_bar = st.progress(0)
                    success_count = 0
                    
                    for i, vehicle in enumerate(plain_vehicles):
                        try:
                            if convert_to_encrypted(vehicle.vehicle_uuid):
                                success_count += 1
                        except:
                            pass
                        
                        progress_bar.progress((i + 1) / len(plain_vehicles))
                    
                    st.success(f"✅ Конвертирани {success_count} от {len(plain_vehicles)} превозни средства!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Грешка: {e}")

with col_bulk2:
    st.markdown("#### Database Maintenance")
    
    if st.button("🔄 Refresh Statistics", type="secondary"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    st.warning("⚠️ Danger Zone")
    confirm_delete_all = st.checkbox("Потвърдете изтриване на ВСИЧКИ данни")
    
    if st.button(
        "🗑️ Delete All Vehicles",
        type="secondary",
        disabled=not confirm_delete_all
    ):
        st.error("⚠️ Тази функция не е имплементирана за безопасност")
        st.info("💡 За да изчистите базата данни, използвайте SQL:")
        st.code("DELETE FROM vehicles;")

st.markdown("---")

# Footer
st.markdown("### 💡 Tips")

st.markdown("""
- **Pagination:** Използвайте page size и page number за навигация
- **Filters:** Комбинирайте филтри за по-точни резултати
- **Bulk Convert:** Конвертирайте всички plain записи в encrypted наведнъж
- **Download:** Изтеглете данните като CSV за backup
- **Visualizations:** Проследявайте разпределението на данните
""")