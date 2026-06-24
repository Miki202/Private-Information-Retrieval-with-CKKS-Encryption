"""
Популиране на базата данни директно от готовите analysis файлове.
Използва вече изчислените embeddings и метаданни.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Добавяне на project root към path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from storage.database.operations import insert_vehicle

def populate_from_analysis(csv_path: Path, npy_path: Path, limit: int = None):
    print(f"Зареждане на метаданни от: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Зареждане на embeddings от: {npy_path}")
    embeddings = np.load(npy_path)
    
    if len(df) != len(embeddings):
        print(f"ГРЕШКА: Разминаване! CSV има {len(df)} реда, а NPY има {len(embeddings)} вектора.")
        sys.exit(1)
        
    if limit:
        df = df.head(limit)
        
    print(f"Ще бъдат обработени {len(df)} записа.")
    print("="*60)
    
    success = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Вмъкване в DB"):
        try:
            # 1. Вземане на метаданни
            # В CSV-то табелите са записани като JSON string: '["AA1234BB"]'
            import json
            plates_str = row.get('plates', '[]')
            try:
                plates_list = json.loads(plates_str)
                license_plate = plates_list[0] if plates_list else None
            except:
                license_plate = None
                
            color = str(row['color_name']) if pd.notna(row.get('color_name')) else None
            image_path = str(row['path'])
            
            # 2. Вземане на съответния embedding
            emb_idx = int(row['embedding_idx'])
            embedding = embeddings[emb_idx]
            
            # Проверка дали embedding-ът не е празен (всички нули)
            if np.all(embedding == 0):
                failed += 1
                continue
            
            # 3. Вмъкване в базата данни (Криптирането става вътре)
            uuid, db_id = insert_vehicle(
                embedding=embedding,
                license_plate=license_plate,
                color=color,
                body_type=None,  # Нямаме body_type в analysis_results
                image_path=image_path
            )
            
            success += 1
            
        except Exception as e:
            print(f"\n✗ Грешка при запис {idx}: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"ГОТОВО! Успешни: {success}, Неуспешни: {failed}")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Използване: python scripts/populate_from_analysis.py <път_до_csv> <път_до_npy> [лимит]")
        print("Пример: python scripts/populate_from_analysis.py analysis_results.csv analysis_embeddings.npy 100")
        sys.exit(1)
        
    csv_file = Path(sys.argv[1])
    npy_file = Path(sys.argv[2])
    limit_val = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if not csv_file.exists():
        print(f"Грешка: CSV файлът {csv_file} не съществува!")
        sys.exit(1)
        
    if not npy_file.exists():
        print(f"Грешка: NPY файлът {npy_file} не съществува!")
        sys.exit(1)
        
    populate_from_analysis(csv_file, npy_file, limit_val)