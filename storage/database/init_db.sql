-- Активиране на pgvector extension за векторно търсене
CREATE EXTENSION IF NOT EXISTS vector;

-- Таблица 1: Метаданни за превозни средства
CREATE TABLE IF NOT EXISTS vehicles (
    id SERIAL PRIMARY KEY,
    vehicle_uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    
    -- Метаданни от pipeline-а на твоя колега
    license_plate VARCHAR(50),  -- Номер на колата
    color VARCHAR(50),           -- Цвят
    body_type VARCHAR(50),       -- Тип на каросерията
    image_path TEXT,             -- Път до снимката
    
    -- Статус на криптирането
    is_encrypted BOOLEAN DEFAULT FALSE,
    
    -- Времеви печати
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица 2: Векторни embeddings (обикновени и криптирани)
CREATE TABLE IF NOT EXISTS vehicle_vectors (
    id SERIAL PRIMARY KEY,
    vehicle_id INTEGER UNIQUE REFERENCES vehicles(id) ON DELETE CASCADE,
    
    -- Обикновен embedding (256 измерения)
    embedding vector(256),
    
    -- Криптиран embedding (съхранен като binary)
    encrypted_embedding BYTEA,
    
    -- Метаданни за криптирането
    encryption_context BYTEA,  -- CKKS контекст
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индекси за бързо търсене по метаданни
CREATE INDEX IF NOT EXISTS idx_license_plate ON vehicles(license_plate);
CREATE INDEX IF NOT EXISTS idx_color ON vehicles(color);
CREATE INDEX IF NOT EXISTS idx_body_type ON vehicles(body_type);
CREATE INDEX IF NOT EXISTS idx_is_encrypted ON vehicles(is_encrypted);

-- Индекс за векторно търсене (cosine similarity) - само за некриптирани
CREATE INDEX IF NOT EXISTS idx_vector_cosine ON vehicle_vectors 
    USING hnsw (embedding vector_cosine_ops) 
    WHERE embedding IS NOT NULL;

-- Функция за автоматично обновяване на timestamp
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger за автоматично обновяване
DROP TRIGGER IF EXISTS update_vehicles_timestamp ON vehicles;
CREATE TRIGGER update_vehicles_timestamp 
    BEFORE UPDATE ON vehicles
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();