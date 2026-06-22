-- TRUE PIR Schema with Encrypted Metadata

CREATE TABLE IF NOT EXISTS vehicles (
    id SERIAL PRIMARY KEY,
    vehicle_uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    
    -- Encryption status
    is_encrypted BOOLEAN DEFAULT FALSE,
    
    -- Plain mode: FAISS ID + visible metadata
    faiss_id INTEGER,
    license_plate VARCHAR(50),
    color VARCHAR(50),
    body_type VARCHAR(50),
    image_path TEXT,
    
    -- TRUE PIR mode: Everything encrypted
    encrypted_embedding BYTEA,           -- CKKS encrypted 256-dim vector (~100KB)
    encrypted_metadata BYTEA,            -- Encrypted JSON with all metadata (~50KB)
    encryption_context BYTEA,            -- CKKS context (~50KB)
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for plain mode only
CREATE INDEX IF NOT EXISTS idx_license_plate ON vehicles(license_plate) WHERE is_encrypted = FALSE;
CREATE INDEX IF NOT EXISTS idx_color ON vehicles(color) WHERE is_encrypted = FALSE;
CREATE INDEX IF NOT EXISTS idx_body_type ON vehicles(body_type) WHERE is_encrypted = FALSE;
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

-- Comments
COMMENT ON TABLE vehicles IS 'Vehicle storage with dual mode: Plain (FAISS) and TRUE PIR (encrypted)';
COMMENT ON COLUMN vehicles.is_encrypted IS 'FALSE = plain mode (FAISS), TRUE = TRUE PIR (fully encrypted)';
COMMENT ON COLUMN vehicles.encrypted_metadata IS 'Encrypted JSON: {license_plate, color, body_type, image_path}';
COMMENT ON COLUMN vehicles.license_plate IS 'Only used in plain mode (NULL for encrypted vehicles)';
COMMENT ON COLUMN vehicles.color IS 'Only used in plain mode (NULL for encrypted vehicles)';
COMMENT ON COLUMN vehicles.body_type IS 'Only used in plain mode (NULL for encrypted vehicles)';