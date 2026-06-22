CREATE TABLE vehicles (
    id SERIAL PRIMARY KEY,
    vehicle_uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),

    license_plate VARCHAR(50),
    color VARCHAR(50),
    body_type VARCHAR(50),
    image_path TEXT,

    encrypted_embedding BYTEA NOT NULL,
    encrypted_metadata BYTEA NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индекси за филтриране по метаданни
CREATE INDEX IF NOT EXISTS idx_license_plate ON vehicles(license_plate);
CREATE INDEX IF NOT EXISTS idx_color ON vehicles(color);
CREATE INDEX IF NOT EXISTS idx_body_type ON vehicles(body_type);

-- Автоматично обновяване на updated_at
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
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

-- Документация за колоните
COMMENT ON COLUMN vehicles.encrypted_embedding
IS 'CKKS encrypted 256-dim vector stored as BYTEA';