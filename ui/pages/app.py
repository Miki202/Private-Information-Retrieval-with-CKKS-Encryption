"""
Главна страница на Streamlit приложението
"""
import streamlit as st
import sys
import os

# Добавяме parent директория
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database.operations import get_database_stats, get_encryption_stats

# Конфигурация на страницата
st.set_page_config(
    page_title="PIR Vehicle Storage",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔐 PIR Vehicle Storage System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Private Information Retrieval with CKKS Encryption</div>', unsafe_allow_html=True)

st.markdown("---")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 🎯 Какво Прави Тази Система?
    
    Това е **Private Information Retrieval (PIR)** система за търсене на превозни средства:
    
    1. **Upload** 📤 - Качване на снимки на коли (plain или криптирано)
    2. **Search** 🔍 - Търсене на подобни коли
        - **Plain Mode**: Бързо, но сървърът вижда query-то
        - **PIR Mode**: Бавно, но query-то е напълно скрито!
    3. **Browse** 📊 - Преглед на всички съхранени превозни средства
    
    ### 🔒 Какво Е PIR?
    
    **Private Information Retrieval** е криптографска техника, която позволява на клиент да
    търси в база данни **без сървърът да знае какво търси**.
    
    - ✅ Query се криптира преди изпращане
    - ✅ Сървърът прави homomorphic операции върху криптирани данни
    - ✅ Резултатите се връщат криптирани
    - ✅ Само клиентът може да декриптира
    """)

with col2:
    st.markdown("### 🔐 PIR Process")
    st.image("https://via.placeholder.com/300x400.png?text=PIR+Diagram", use_column_width=True)
    # Можеш да създадеш истинска диаграма по-късно

st.markdown("---")

# Database Statistics
st.markdown("### 📊 Database Statistics")

try:
    general_stats = get_database_stats()
    encryption_stats = get_encryption_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🚗 Total Vehicles",
            value=general_stats['total_vehicles'],
            help="Общ брой превозни средства в базата данни"
        )
    
    with col2:
        st.metric(
            label="🔓 Plain Mode",
            value=general_stats['plain_count'],
            help="Брой превозни средства в plain mode"
        )
    
    with col3:
        st.metric(
            label="🔐 Encrypted (PIR)",
            value=general_stats['encrypted_count'],
            help="Брой превозни средства в encrypted mode"
        )
    
    with col4:
        encryption_pct = encryption_stats['encryption_percentage']
        st.metric(
            label="📈 Encryption Rate",
            value=f"{encryption_pct:.1f}%",
            help="Процент криптирани записи"
        )
    
    # Detailed stats
    with st.expander("📈 Detailed Statistics"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Color Distribution:**")
            if general_stats['color_distribution']:
                for color, count in general_stats['color_distribution'].items():
                    st.text(f"  {color}: {count}")
            else:
                st.text("  No data")
        
        with col_b:
            st.markdown("**Body Type Distribution:**")
            if general_stats['body_type_distribution']:
                for body_type, count in general_stats['body_type_distribution'].items():
                    st.text(f"  {body_type}: {count}")
            else:
                st.text("  No data")
        
        st.markdown("**Storage Stats:**")
        st.text(f"  Avg encrypted size: {encryption_stats['avg_encrypted_size_bytes']:,} bytes")
        st.text(f"  Avg context size: {encryption_stats['avg_context_size_bytes']:,} bytes")
        st.text(f"  Overhead: {encryption_stats['avg_total_size_bytes'] / max(encryption_stats['plain_embedding_size_bytes'], 1):.1f}x")

except Exception as e:
    st.error(f"❌ Грешка при зареждане на статистика: {e}")
    st.info("💡 Уверете се че базата данни е инициализирана и връзката работи")

st.markdown("---")

# Features
st.markdown("### ✨ Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h4>📤 Upload Modes</h4>
        <ul>
            <li><b>Plain</b>: Бързо качване</li>
            <li><b>Encrypted</b>: PIR-ready криптиране</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h4>🔍 Search Modes</h4>
        <ul>
            <li><b>Plain</b>: ~10ms търсене</li>
            <li><b>PIR</b>: ~500ms, напълно private</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
        <h4>🎯 Filtering</h4>
        <ul>
            <li>По цвят</li>
            <li>По тип каросерия</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Navigation
st.markdown("### 🧭 Navigation")
st.info("👈 Използвай sidebar-a за да навигираш между страниците")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📤 Upload")
    st.markdown("Качване на нови превозни средства")
    
with col2:
    st.markdown("#### 🔍 Search")
    st.markdown("Търсене на подобни превозни средства")

with col3:
    st.markdown("#### 📊 Browse")
    st.markdown("Преглед на всички записи")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🔐 Built with CKKS Homomorphic Encryption | PostgreSQL + pgvector | Streamlit</p>
    <p>Private Information Retrieval Project</p>
</div>
""", unsafe_allow_html=True)