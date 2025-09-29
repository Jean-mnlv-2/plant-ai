-- Script d'initialisation de la base de données PostgreSQL pour Plant-AI
-- Ce script est exécuté automatiquement lors de la création du conteneur

-- Créer l'extension pour les statistiques avancées
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Créer l'extension pour les fonctions JSON avancées
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Configuration des paramètres pour les performances
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET track_activity_query_size = 2048;
ALTER SYSTEM SET log_statement = 'mod';
ALTER SYSTEM SET log_min_duration_statement = 1000;

-- Créer un utilisateur dédié pour les backups
CREATE USER plant_ai_backup WITH PASSWORD 'backup_secure_password_2024';
GRANT CONNECT ON DATABASE plant_ai TO plant_ai_backup;
GRANT USAGE ON SCHEMA public TO plant_ai_backup;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO plant_ai_backup;

-- Créer un utilisateur pour les analytics (lecture seule)
CREATE USER plant_ai_analytics WITH PASSWORD 'analytics_secure_password_2024';
GRANT CONNECT ON DATABASE plant_ai TO plant_ai_analytics;
GRANT USAGE ON SCHEMA public TO plant_ai_analytics;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO plant_ai_analytics;

-- Configuration des permissions
GRANT ALL PRIVILEGES ON DATABASE plant_ai TO plant_ai;
GRANT ALL PRIVILEGES ON SCHEMA public TO plant_ai;

-- Créer des vues pour les analytics
CREATE OR REPLACE VIEW daily_predictions AS
SELECT 
    DATE(timestamp) as prediction_date,
    COUNT(*) as total_predictions,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(confidence_avg) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time,
    SUM(num_detections) as total_detections
FROM predictions
GROUP BY DATE(timestamp)
ORDER BY prediction_date DESC;

CREATE OR REPLACE VIEW top_diseases AS
SELECT 
    class_name,
    detection_count,
    last_detected,
    ROUND(detection_count::numeric / (SELECT SUM(detection_count) FROM detected_classes) * 100, 2) as percentage
FROM detected_classes
ORDER BY detection_count DESC;

CREATE OR REPLACE VIEW user_activity AS
SELECT 
    user_id,
    first_seen,
    last_seen,
    total_predictions,
    total_detections,
    avg_confidence,
    EXTRACT(EPOCH FROM (last_seen - first_seen))/3600 as hours_active
FROM users
WHERE is_active = true
ORDER BY total_predictions DESC;

-- Créer des fonctions utiles
CREATE OR REPLACE FUNCTION cleanup_old_data(days_to_keep INTEGER DEFAULT 30)
RETURNS TABLE(deleted_predictions BIGINT, deleted_metrics BIGINT) AS $$
DECLARE
    pred_count BIGINT;
    metrics_count BIGINT;
BEGIN
    -- Supprimer les anciennes prédictions
    DELETE FROM predictions 
    WHERE timestamp < NOW() - INTERVAL '1 day' * days_to_keep;
    GET DIAGNOSTICS pred_count = ROW_COUNT;
    
    -- Supprimer les anciennes métriques
    DELETE FROM performance_metrics 
    WHERE timestamp < NOW() - INTERVAL '1 day' * days_to_keep;
    GET DIAGNOSTICS metrics_count = ROW_COUNT;
    
    -- VACUUM pour optimiser
    VACUUM ANALYZE;
    
    RETURN QUERY SELECT pred_count, metrics_count;
END;
$$ LANGUAGE plpgsql;

-- Créer des index pour les performances
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp_brin ON predictions USING BRIN (timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_user_timestamp ON predictions (user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_endpoint_timestamp ON performance_metrics (endpoint, timestamp DESC);

-- Configuration des statistiques
ANALYZE;

-- Message de confirmation
DO $$
BEGIN
    RAISE NOTICE 'Base de données Plant-AI initialisée avec succès !';
    RAISE NOTICE 'Extensions installées: pg_stat_statements, btree_gin';
    RAISE NOTICE 'Vues créées: daily_predictions, top_diseases, user_activity';
    RAISE NOTICE 'Fonction de nettoyage: cleanup_old_data()';
END $$;
