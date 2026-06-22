#!/usr/bin/env python3
"""
Скрипт за инициализация на системата
"""

import subprocess
import sys
import os

def print_header(text):
    print("\n" + "="*60)
    print(text)
    print("="*60)

def run_command(cmd, check=True):
    """Изпълнява команда"""
    print(f"\n→ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0 and check:
        print(f"✗ Грешка: {result.stderr}")
        return False
    
    if result.stdout:
        print(result.stdout)
    
    print("✓ Успешно")
    return True

def check_postgres():
    """Проверява PostgreSQL инсталацията"""
    print_header("1. Проверка на PostgreSQL")
    
    # Windows
    if sys.platform == "win32":
        result = subprocess.run("where psql", shell=True, capture_output=True)
        if result.returncode != 0:
            print("✗ PostgreSQL не е намерен!")
            print("\nИнструкции за Windows:")
            print("1. Изтегли от: https://www.postgresql.org/download/windows/")
            print("2. Инсталирай PostgreSQL 16")
            print("3. Добави в PATH: C:\\Program Files\\PostgreSQL\\16\\bin")
            return False
    else:
        # Linux/Mac
        result = subprocess.run("which psql", shell=True, capture_output=True)
        if result.returncode != 0:
            print("✗ PostgreSQL не е намерен!")
            print("\nИнсталирай с:")
            print("  Ubuntu: sudo apt install postgresql postgresql-contrib")
            print("  Mac: brew install postgresql@16")
            return False
    
    print("✓ PostgreSQL е инсталиран")
    return True

def create_database():
    """Създава базата данни"""
    print_header("2. Създаване на база данни")
    
    # Проверка дали съществува
    check_cmd = 'psql -U postgres -lqt | cut -d "|" -f 1 | grep -qw vehicle_storage'
    result = subprocess.run(check_cmd, shell=True, capture_output=True)
    
    if result.returncode == 0:
        print("✓ База данни vehicle_storage вече съществува")
        return True
    
    # Създаване
    if sys.platform == "win32":
        cmd = 'psql -U postgres -c "CREATE DATABASE vehicle_storage"'
    else:
        cmd = 'createdb vehicle_storage'
    
    return run_command(cmd, check=False)

def init_schema():
    """Инициализира схемата"""
    print_header("3. Инициализация на схема")
    
    sql_file = "database/init_db.sql"
    
    if not os.path.exists(sql_file):
        print(f"✗ Файлът {sql_file} не е намерен!")
        return False
    
    cmd = f'psql -U postgres -d vehicle_storage -f {sql_file}'
    return run_command(cmd)

def test_connection():
    """Тества връзката"""
    print_header("4. Тестване на връзката")
    
    try:
        from database.connection import test_connection
        return test_connection()
    except Exception as e:
        print(f"✗ Грешка: {e}")
        return False

def install_python_deps():
    """Инсталира Python зависимости"""
    print_header("5. Инсталация на Python зависимости")
    
    if not os.path.exists("requirements.txt"):
        print("✗ requirements.txt не е намерен!")
        return False
    
    cmd = f"{sys.executable} -m pip install -r requirements.txt"
    return run_command(cmd)

def main():
    print_header("🚀 PIR Vehicle Storage System - Setup")
    
    print("\nТози скрипт ще конфигурира системата стъпка по стъпка.")
    input("Натиснете Enter за да продължите...")
    
    # Стъпки
    steps = [
        ("PostgreSQL", check_postgres),
        ("Create Database", create_database),
        ("Initialize Schema", init_schema),
        ("Python Dependencies", install_python_deps),
        ("Test Connection", test_connection),
    ]
    
    results = {}
    
    for name, func in steps:
        try:
            results[name] = func()
        except Exception as e:
            print(f"\n✗ Грешка при {name}: {e}")
            results[name] = False
    
    # Summary
    print_header("📊 Резюме")
    
    all_success = True
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {name}")
        if not success:
            all_success = False
    
    if all_success:
        print_header("✅ Setup завършен успешно!")
        print("\nСледващи стъпки:")
        print("1. Стартирай Streamlit: streamlit run ui/app.py")
        print("2. Отвори браузър: http://localhost:8501")
        print("3. Качи превозни средства от Upload страницата")
        print("4. Тествай PIR търсене от Search страницата")
    else:
        print_header("⚠️ Setup завърши с грешки")
        print("\nПроверете грешките по-горе и опитайте отново.")
        print("За помощ: проверете README.md")

if __name__ == "__main__":
    main()