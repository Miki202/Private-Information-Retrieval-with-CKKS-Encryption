-- TRUE PIR Schema
-- Only metadata in PostgreSQL, encrypted vectors stored inline

CREATE TABLE IF NOT EXISTS vehicles (
    id SERIAL PRIMARY KEY,
    vehicle_uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    
    -- Метаданни (видими за server)
    license_plate VARCHAR(50),
    color VARCHAR(50),
    body_type VARCHAR(50),
    image_path TEXT,
    
    -- Encryption status
    is_encrypted BOOLEAN DEFAULT FALSE,
    
    -- Plain mode: FAISS ID (за comparison baseline)
    faiss_id INTEGER,
    
    -- TRUE PIR mode: Encrypted data stored in DB
    encrypted_embedding BYTEA,      -- CKKS encrypted 256-dim vector
    encryption_context BYTEA,       -- CKKS context (includes keys for now)
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индекси за metadata filtering
CREATE INDEX IF NOT EXISTS idx_license_plate ON vehicles(license_plate);
CREATE INDEX IF NOT EXISTS idx_color ON vehicles(color);
CREATE INDEX IF NOT EXISTS idx_body_type ON vehicles(body_type);
CREATE INDEX IF NOT EXISTS idx_is_encrypted ON vehicles(is_encrypted);
CREATE INDEX IF NOT EXISTS idx_faiss_id ON vehicles(faiss_id);

-- Timestamp auto-update
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_vehicles_timestamp ON vehicles;
CREATE TRIGGER update_vehicles_timestamp 
    BEFORE UPDATE ON vehicles
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

-- Comments for documentation
COMMENT ON COLUMN vehicles.encrypted_embedding IS 'CKKS encrypted 256-dim vector (BYTEA, ~100KB)';
COMMENT ON COLUMN vehicles.encryption_context IS 'CKKS context with keys (BYTEA, ~50KB). In production, server should only have public key.';
COMMENT ON COLUMN vehicles.is_encrypted IS 'TRUE = PIR mode (homomorphic search), FALSE = plain mode (FAISS search)';