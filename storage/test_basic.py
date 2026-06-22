"""
Тестов скрипт за проверка на основните операции
"""
import numpy as np
from database.operations import (
    insert_vehicle,
    search_similar,
    get_all_vehicles,
    get_vehicle_by_uuid,
    delete_vehicle,
    get_database_stats
)

def test_basic_operations():
    print("=" * 50)
    print("ТЕСТВАНЕ НА ОСНОВНИ ОПЕРАЦИИ")
    print("=" * 50)
    
    # Тест 1: Вмъкване на превозно средство
    print("\n1. Тест на вмъкване...")
    embedding1 = np.random.rand(256)  # Генерираме случаен embedding
    uuid1, id1 = insert_vehicle(
        license_plate="TEST123",
        color="red",
        body_type="sedan",
        embedding=embedding1,
        image_path="test_image.jpg"
    )
    print(f"   Създадено: UUID={uuid1}, ID={id1}")
    
    # Тест 2: Вмъкване на още превозни средства
    print("\n2. Вмъкване на още 2 превозни средства...")
    embedding2 = np.random.rand(256)
    uuid2, id2 = insert_vehicle(
        license_plate="TEST456",
        color="blue",
        body_type="suv",
        embedding=embedding2
    )
    
    embedding3 = np.random.rand(256)
    uuid3, id3 = insert_vehicle(
        license_plate="TEST789",
        color="red",
        body_type="coupe",
        embedding=embedding3
    )
    
    # Тест 3: Извличане на всички
    print("\n3. Извличане на всички превозни средства...")
    vehicles = get_all_vehicles()
    print(f"   Намерени {len(vehicles)} превозни средства")
    for v in vehicles:
        print(f"   - {v.license_plate} ({v.color} {v.body_type})")
    
    # Тест 4: Извличане по UUID
    print(f"\n4. Извличане по UUID: {uuid1}...")
    vehicle = get_vehicle_by_uuid(uuid1)
    if vehicle:
        print(f"   Намерено: {vehicle.license_plate}")
    
    # Тест 5: Търсене на подобни
    print("\n5. Търсене на подобни превозни средства...")
    query_embedding = embedding1 + np.random.rand(256) * 0.1  # Малко различен от embedding1
    results = search_similar(query_embedding, top_k=3)
    print(f"   Намерени {len(results)} резултата:")
    for i, result in enumerate(results):
        v = result["vehicle"]
        score = result["similarity_score"]
        print(f"   {i+1}. {v['license_plate']} - Similarity: {score:.4f}")
    
    # Тест 6: Търсене с филтър
    print("\n6. Търсене само за червени коли...")
    results = search_similar(query_embedding, top_k=5, filter_color="red")
    print(f"   Намерени {len(results)} червени коли")
    
    # Тест 7: Статистика
    print("\n7. Статистика на базата данни...")
    stats = get_database_stats()
    print(f"   Общо превозни средства: {stats['total_vehicles']}")
    print(f"   Общо вектори: {stats['total_vectors']}")
    print(f"   Криптирани: {stats['encrypted_count']}")
    print(f"   Некриптирани: {stats['plain_count']}")
    print(f"   Разпределение по цветове: {stats['color_distribution']}")
    
    # Тест 8: Изтриване
    print(f"\n8. Изтриване на превозно средство {uuid3}...")
    success = delete_vehicle(uuid3)
    if success:
        print("   Изтрито успешно")
    
    # Проверка след изтриване
    vehicles = get_all_vehicles()
    print(f"   Останали превозни средства: {len(vehicles)}")
    
    print("\n" + "=" * 50)
    print("✓ ВСИЧКИ ТЕСТОВЕ ЗАВЪРШИХА")
    print("=" * 50)

if __name__ == "__main__":
    test_basic_operations()