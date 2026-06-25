import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from storage.database.operations import insert_vehicle

def populate_from_analysis(csv_path: Path, npy_path: Path, limit: int = None):
    print(f"Зареждане на метаданни от: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Зареждане на embeddings от: {npy_path}")
    embeddings = np.load(npy_path)
    
    if len(df) != len(embeddings):
        sys.exit(1)
        
    if limit:
        df = df.head(limit)
    success = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Вмъкване в DB"):
        try:
            import json
            plates_str = row.get('plates', '[]')
            try:
                plates_list = json.loads(plates_str)
                license_plate = plates_list[0] if plates_list else None
            except:
                license_plate = None
                
            color = str(row['color_name']) if pd.notna(row.get('color_name')) else None
            image_path = str(row['path'])
            emb_idx = int(row['embedding_idx'])
            embedding = embeddings[emb_idx]
            if np.all(embedding == 0):
                failed += 1
                continue

            uuid, db_id = insert_vehicle(
                embedding=embedding,
                license_plate=license_plate,
                color=color,
                body_type=None,  
                image_path=image_path
            )
            
            success += 1
        except Exception as e:
            print(f"\nГрешка: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"Успешни: {success}, Неуспешни: {failed}")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
        
    csv_file = Path(sys.argv[1])
    npy_file = Path(sys.argv[2])
    limit_val = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if not csv_file.exists():
        print(f"Грешка: CSV файлът {csv_file} го няма")
        sys.exit(1)
        
    if not npy_file.exists():
        print(f"Грешка: NPY файлът {npy_file} го няма.")
        sys.exit(1)
        
    populate_from_analysis(csv_file, npy_file, limit_val)