import time
import os
import sys
import numpy as np
import tenseal as ts
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from storage.database.connection import get_db
from storage.database.models import Vehicle
from storage.database.encryption import (
    get_context,
    decrypt_embedding_simple,
    decrypt_metadata
)

NET_CLIENT_SERVER_ONE_WAY = 8.0  
NET_SERVER_SERVER_RTT = 4.0      

def run_whole_system_benchmark():
    db = get_db()
    
    try:
        vehicles = db.query(Vehicle).all()
        num_records = len(vehicles)
        
        if num_records == 0:
            print("Базата данни е празна.")
            return

        real_embeddings = [decrypt_embedding_simple(v.encrypted_embedding) for v in vehicles]
        db_plain = np.array(real_embeddings)
        DIMENSIONS = db_plain.shape[1]
        query_plain = db_plain[0] 
        context = get_context()
        start_ckks_client_prep = time.time()
        real_embeddings = [decrypt_embedding_simple(v.encrypted_embedding) for v in vehicles]
        db_plain = np.array(real_embeddings)
        DIMENSIONS = db_plain.shape[1]
        query_plain = db_plain[0] 
        
        context = get_context()
        flat_query = np.array(query_plain).flatten()
        cleaned_query = np.clip(flat_query, -1.0, 1.0)
        query_ready_for_ts = [float(x) for x in cleaned_query]

        start_ckks_client_prep = time.time()
        encrypted_query = ts.ckks_vector(context, query_ready_for_ts)
        ckks_phase1 = ((time.time() - start_ckks_client_prep) * 1000) + NET_CLIENT_SERVER_ONE_WAY
        start_ckks_server = time.time()
        ckks_encrypted_results = []
        for v in vehicles:
            enc_row = ts.ckks_vector_from(context, v.encrypted_embedding)
            res = encrypted_query.dot(enc_row)
            ckks_encrypted_results.append(res)
        ckks_phase2 = (time.time() - start_ckks_server) * 1000
        start_ckks_client_dec = time.time()
        ckks_final_scores = [float(res.decrypt()[0]) for res in ckks_encrypted_results]
        
        K = min(20, num_records)
        top_k_indices = np.argsort(ckks_final_scores)[-K:][::-1]
        for idx in top_k_indices:
            _ = decrypt_metadata(vehicles[idx].encrypted_metadata)
        
        ckks_phase3 = ((time.time() - start_ckks_client_dec) * 1000) + NET_CLIENT_SERVER_ONE_WAY
        ckks_total = ckks_phase1 + ckks_phase2 + ckks_phase3
        
        db_share1 = np.random.randn(num_records, DIMENSIONS)
        db_share2 = db_plain - db_share1

        start_mpc_client_prep = time.time()
        query_share1 = np.random.randn(DIMENSIONS)
        query_share2 = query_plain - query_share1
        mpc_phase1 = ((time.time() - start_mpc_client_prep) * 1000) + NET_CLIENT_SERVER_ONE_WAY
        start_mpc_server = time.time()
        server_a_results = np.dot(db_share1, query_share1)
        server_b_results = np.dot(db_share2, query_share2)
        server_a_cross = np.dot(db_share1, query_share2)
        server_b_cross = np.dot(db_share2, query_share1)
        server_a_total = server_a_results + server_a_cross
        server_b_total = server_b_results + server_b_cross
        mpc_phase2 = ((time.time() - start_mpc_server) * 1000) + NET_SERVER_SERVER_RTT
        start_mpc_client_dec = time.time()
        mpc_final_scores = server_a_total + server_b_total
        
        for _ in range(K):
            _ = "Симулирана реконструирана текстова информация (XOR)" 
        
        mpc_phase3 = ((time.time() - start_mpc_client_dec) * 1000) + NET_CLIENT_SERVER_ONE_WAY
        mpc_total = mpc_phase1 + mpc_phase2 + mpc_phase3

        print("\nДОКЛАД:")
        print("=" * 80)
        print(f"| Фаза от жизнения цикъл      | Архитектура А (CKKS)  | Архитектура Б (MPC)        |")
        print(f"|-----------------------------|-----------------------|----------------------------|")
        print(f"| 1. Клиент: Подготовка + Път | {ckks_phase1:18.2f} ms | {mpc_phase1:23.2f} ms |")
        print(f"| 2. Сървър: Компютърен Трафик| {ckks_phase2:18.2f} ms | {mpc_phase2:23.2f} ms |")
        print(f"| 3. Клиент: Резолюция + Връща| {ckks_phase3:18.2f} ms | {mpc_phase3:23.2f} ms |")
        print(f"|-----------------------------|-----------------------|----------------------------|")
        print(f"| ОБЩО ВРЕМЕ   | {ckks_total:18.2f} ms | {mpc_total:23.2f} ms |")
        print("=" * 80)

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 10
        
        color_ckks = '#1A365D'
        color_mpc = '#5A738E'  
        colors = [color_ckks, color_mpc]
        labels = ['Архитектура А (CKKS)', 'Архитектура Б (MPC)']

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Времеви анализ на защитените системи с включен мрежов overhead\n(База данни: N = {num_records}, Оптичен интернет до облака, Резултати: Top-K = {K})', fontsize=12, fontweight='bold', y=0.96)

        def style_subplot(ax, title, data, y_label="Общо време (ms)"):
            bars = ax.bar(labels, data, color=colors, width=0.4, edgecolor='#718096', linewidth=0.5)
            ax.set_title(title, fontsize=11, fontweight='bold', pad=10, color='#2D3748')
            ax.set_ylabel(y_label, fontsize=10, labelpad=5)
            ax.grid(axis='y', linestyle=':', linewidth=0.5, color='#CBD5E0', alpha=0.7)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#718096')
            ax.spines['bottom'].set_color('#718096')
            
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f} ms',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='semibold', color='#2D3748')

        style_subplot(axs[0, 0], "Фаза 1: Мрежово подаване + Подготовка на заявката", [ckks_phase1, mpc_phase1])
        style_subplot(axs[0, 1], "Фаза 2: Изчисления на сървъра (+ Междусървърна мрежа за MPC)", [ckks_phase2, mpc_phase2])
        style_subplot(axs[1, 0], "Фаза 3: Мрежово получаване на отговора + Декриптиране/Реконструкция", [ckks_phase3, mpc_phase3])
        style_subplot(axs[1, 1], "Общо време на жизнения цикъл (End-to-End)", [ckks_total, mpc_total])

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        output_dir = os.path.join("artifacts", "pngs")
        os.makedirs(output_dir, exist_ok=True)  
        
        chart_name = os.path.join(output_dir, "system_e2e_comparison.png")
        plt.savefig(chart_name, dpi=300, bbox_inches='tight')
        plt.close()
    finally:
        db.close()

if __name__ == "__main__":
    run_whole_system_benchmark()