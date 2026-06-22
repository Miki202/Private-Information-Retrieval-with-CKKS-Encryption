#!/usr/bin/env python3
"""
Setup script за PostgreSQL + FAISS system
"""
import subprocess
import sys
import os

def print_header(text):
    print("\n" + "="*60)
    print(text)
    print("="*60)

def check_postgres():
    """Проверка на PostgreSQL"""
    print_header("1. PostgreSQL Check")
    
    if sys.platform == "win32":
        result = subprocess.run("where psql", shell=True, capture_output=True)
    else:
        result = subprocess.run("which psql", shell=True, capture_output=True)
    
    if result.returncode != 0:
        print("✗ PostgreSQL not found!")
        print("\nInstall from: https://www.postgresql.org/download/")
        return False
    
    print("✓ PostgreSQL found")
    return True

def create_database():
    """Създава database"""
    print_header("2. Create Database")
    
    cmd = 'psql -U postgres -c "CREATE DATABASE vehicle_storage"'
    result = subprocess.run(cmd, shell=True, capture_output=True)
    
    if b"already exists" in result.stderr:
        print("✓ Database already exists")
        return True
    elif result.returncode == 0:
        print("✓ Database created")
        return True
    else:
        print(f"✗ Error: {result.stderr.decode()}")
        return False

def init_schema():
    """Инициализира schema"""
    print_header("3. Initialize Schema")
    
    sql_file = "database/init_db.sql"
    if not os.path.exists(sql_file):
        print(f"✗ {sql_file} not found")
        return False
    
    cmd = f'psql -U postgres -d vehicle_storage -f {sql_file}'
    result = subprocess.run(cmd, shell=True, capture_output=True)
    
    if result.returncode == 0:
        print("✓ Schema initialized")
        return True
    else:
        print(f"✗ Error: {result.stderr.decode()}")
        return False

def install_deps():
    """Инсталира Python dependencies"""
    print_header("4. Install Python Dependencies")
    
    cmd = f"{sys.executable} -m pip install -r requirements.txt"
    result = subprocess.run(cmd, shell=True)
    
    return result.returncode == 0

def test_system():
    """Тества системата"""
    print_header("5. Test System")
    
    try:
        from database.connection import test_connection
        from database.vector_store import get_plain_store, get_encrypted_store
        
        # Test PostgreSQL
        if not test_connection():
            return False
        
        # Test FAISS
        plain = get_plain_store()
        encrypted = get_encrypted_store()
        print(f"✓ FAISS plain: {plain.get_stats()['total_vectors']} vectors")
        print(f"✓ FAISS encrypted: {encrypted.get_stats()['total_vectors']} vectors")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print_header("🚀 PostgreSQL + FAISS Setup")
    
    steps = [
        ("PostgreSQL Check", check_postgres),
        ("Create Database", create_database),
        ("Initialize Schema", init_schema),
        ("Install Dependencies", install_deps),
        ("Test System", test_system),
    ]
    
    results = {}
    for name, func in steps:
        try:
            results[name] = func()
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results[name] = False
    
    print_header("Summary")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    if all(results.values()):
        print_header("✅ Setup Complete!")
        print("\nNext steps:")
        print("1. streamlit run ui/app.py")
        print("2. Open: http://localhost:8501")
    else:
        print_header("⚠️ Setup had errors")

if __name__ == "__main__":
    main()