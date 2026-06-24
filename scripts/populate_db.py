"""
Популиране на база данни с пълни профили на автомобили.
Използва ВСИЧКИ модели: encoding, цвят, табели.

Използване: python scripts/populate_db.py <папка_с_изображения> [лимит]
"""
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Добавяне на project root към path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

# Import на всички модели
from notebooks.img2vec_example import encode
from notebooks.car_color_example import extract_color
from notebooks.car_plate_example import detect_plates, ocr_plates_batch

from storage.database.operations import insert_vehicle


def process_image_full_pipeline(image_path: Path) -> dict:
    """
    Пълен 4-етапен pipeline за обработка на автомобилно изображение.
    
    Етапи:
        1. Vehicle encoding (256-D вектор)
        2. Цвят (HSV класификация)
        3. Регистрационна табела (YOLO + CRNN OCR)
        4. Тип на автомобила (TODO)
    
    Returns:
        dict с embedding, license_plate, color, body_type или None при грешка
    """
    try:
        # Зареждане на изображение
        pil_image = Image.open(image_path).convert('RGB')
        
        # ============================================================
        # ЕТАП 1: Vehicle Encoding
        # ============================================================
        print(f'  [1/4] Encoding...', end=' ', flush=True)
        embedding_tensor = encode(pil_image)
        embedding = embedding_tensor.cpu().numpy()
        print(f'✓ shape={embedding.shape}')
        
        # ============================================================
        # ЕТАП 2: Извличане на цвят
        # ============================================================
        print(f'  [2/4] Цвят...', end=' ', flush=True)
        try:
            color_result = extract_color(pil_image)
            # extract_color връща dict като {"name": "red", "share": 0.43, ...}
            color_name = color_result.get('name') if isinstance(color_result, dict) else str(color_result)
            print(f'✓ {color_name}')
        except Exception as e:
            color_name = None
            print(f'✗ ({e})')
        
        # ============================================================
        # ЕТАП 3: Детекция и OCR на табели
        # ============================================================
        print(f'  [3/4] Табела...', end=' ', flush=True)
        try:
            plate_crops = detect_plates(pil_image)
            
            if plate_crops:
                plate_texts = ocr_plates_batch(plate_crops)
                license_plate = plate_texts[0] if plate_texts else None
                print(f'✓ {license_plate}')
            else:
                license_plate = None
                print('⊘ (не е открита)')
        except Exception as e:
            license_plate = None
            print(f'✗ ({e})')
        
        # ============================================================
        # ЕТАП 4: Тип на автомобила (body type)
        # ============================================================
        print(f'  [4/4] Тип...', end=' ', flush=True)
        body_type = None  # TODO: добави vehicle type classifier ако има
        print('⊘ (не е имплементирано)')
        
        # Резултат
        return {
            'embedding': embedding,
            'license_plate': license_plate,
            'color': color_name,
            'body_type': body_type,
            'image_path': str(image_path.resolve())
        }
        
    except Exception as e:
        print(f'  ✗ Фатална грешка: {e}')
        import traceback
        traceback.print_exc()
        return None


def populate_database(image_folder: Path, limit: int = None):
    """
    Обработва всички изображения в папка и ги вкарва в базата.
    
    Args:
        image_folder: Папка с изображения на автомобили
        limit: Максимален брой изображения (None = всички)
    """
    
    # Намиране на всички изображения
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_folder.glob(ext))
    
    if limit:
        image_paths = image_paths[:limit]
    
    if not image_paths:
        print(f'⚠️  Не са намерени изображения в {image_folder}')
        return
    
    print(f'Намерени {len(image_paths)} изображения в {image_folder}')
    print('='*60)
    
    success_count = 0
    fail_count = 0
    
    for img_path in image_paths:
        print(f'\n[{success_count + fail_count + 1}/{len(image_paths)}] {img_path.name}')
        
        # Обработка с пълен pipeline
        profile = process_image_full_pipeline(img_path)
        
        if profile is None:
            fail_count += 1
            continue
        
        # Вмъкване в базата данни
        try:
            print(f'  [DB] Криптиране и вмъкване...', end=' ', flush=True)
            
            uuid, db_id = insert_vehicle(
                embedding=profile['embedding'],
                license_plate=profile['license_plate'],
                color=profile['color'],
                body_type=profile['body_type'],
                image_path=profile['image_path']
            )
            
            print(f'✓ UUID={uuid[:8]}..., ID={db_id}')
            success_count += 1
            
        except Exception as e:
            print(f'✗ Database грешка: {e}')
            fail_count += 1
    
    # Обобщение
    print(f'\n{'='*60}')
    print(f'РЕЗУЛТАТИ:')
    print(f'  ✓ Успешни:    {success_count}')
    print(f'  ✗ Неуспешни:  {fail_count}')
    print(f'  📊 Общо:      {len(image_paths)}')
    
    if len(image_paths) > 0:
        success_rate = success_count / len(image_paths) * 100
        print(f'  💾 Успеваемост: {success_rate:.1f}%')
    
    # Финална статистика на базата
    try:
        from storage.database.operations import get_database_stats
        stats = get_database_stats()
        print(f'\n📦 Общо записи в базата: {stats["total_vehicles"]}')
    except Exception:
        pass


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Използване: python scripts/populate_db.py <папка> [лимит]')
        print('')
        print('Примери:')
        print('  python scripts/populate_db.py test_images/')
        print('  python scripts/populate_db.py pipe/ 5')
        print('  python scripts/populate_db.py C:\\Data\\cars\\ 100')
        sys.exit(1)
    
    folder = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not folder.exists():
        print(f'✗ Грешка: Папката {folder} не съществува')
        sys.exit(1)
    
    if not folder.is_dir():
        print(f'✗ Грешка: {folder} не е папка')
        sys.exit(1)
    
    populate_database(folder, limit)
