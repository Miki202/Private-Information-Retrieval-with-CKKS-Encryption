-- Таблица за метаданни на превозни средства
CREATE TABLE IF NOT EXISTS vehicles (
    id SERIAL PRIMARY KEY,
    vehicle_uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    
    -- Метаданни
    license_plate VARCHAR(50),
    color VARCHAR(50),
    body_type VARCHAR(50),
    image_path TEXT,
    
    -- Encryption status
    is_encrypted BOOLEAN DEFAULT FALSE,
    
    -- FAISS mapping
    faiss_id INTEGER,  -- ID в FAISS index
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индекси за бързо търсене
CREATE INDEX IF NOT EXISTS idx_license_plate ON vehicles(license_plate);
CREATE INDEX IF NOT EXISTS idx_color ON vehicles(color);
CREATE INDEX IF NOT EXISTS idx_body_type ON vehicles(body_type);
CREATE INDEX IF NOT EXISTS idx_is_encrypted ON vehicles(is_encrypted);
CREATE INDEX IF NOT EXISTS idx_faiss_id ON vehicles(faiss_id);

-- Timestamp update функция
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger
DROP TRIGGER IF EXISTS update_vehicles_timestamp ON vehicles;
CREATE TRIGGER update_vehicles_timestamp 
    BEFORE UPDATE ON vehicles
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();