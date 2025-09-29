from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import asyncpg
from .settings import settings

logger = logging.getLogger("plant_ai.database")


class PostgreSQLDatabase:
    """Gestionnaire de base de données PostgreSQL pour Plant-AI Production."""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or self._get_connection_string()
        self.pool: Optional[asyncpg.Pool] = None
    
    def _get_connection_string(self) -> str:
        """Construit la chaîne de connexion PostgreSQL."""
        # Configuration via variables d'environnement
        import os
        
        host = os.getenv("PLANT_AI_DB_HOST", "localhost")
        port = os.getenv("PLANT_AI_DB_PORT", "5432")
        database = os.getenv("PLANT_AI_DB_NAME", "plant_ai")
        user = os.getenv("PLANT_AI_DB_USER", "plant_ai")
        password = os.getenv("PLANT_AI_DB_PASSWORD", "plant_ai_password")
        
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    async def initialize(self):
        """Initialise la connexion et crée les tables."""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            if self.pool is None:
                raise RuntimeError("Failed to create connection pool")
            
            async with self.pool.acquire() as conn:
                await self._create_tables(conn)
                await self._create_indexes(conn)
                
            logger.info("PostgreSQL database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    async def _create_tables(self, conn: asyncpg.Connection):
        """Crée les tables de la base de données."""
        
        # Table des prédictions
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                image_filename VARCHAR(500),
                image_width INTEGER,
                image_height INTEGER,
                predictions_json JSONB NOT NULL,
                processing_time_ms INTEGER,
                confidence_avg REAL,
                num_detections INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Table des métriques de performance
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                endpoint VARCHAR(255) NOT NULL,
                response_time_ms INTEGER NOT NULL,
                status_code INTEGER NOT NULL,
                user_id VARCHAR(255),
                error_message TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Table des statistiques d'utilisation du modèle
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS model_usage (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                model_version VARCHAR(100) NOT NULL,
                predictions_count INTEGER NOT NULL,
                avg_confidence REAL,
                unique_users INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Table des utilisateurs (pour analytics avancées)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) UNIQUE NOT NULL,
                first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                total_predictions INTEGER DEFAULT 0,
                total_detections INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0.0,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Table des classes détectées (pour analytics)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS detected_classes (
                id SERIAL PRIMARY KEY,
                class_name VARCHAR(255) NOT NULL,
                detection_count INTEGER DEFAULT 0,
                last_detected TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def _create_indexes(self, conn: asyncpg.Connection):
        """Crée les index pour optimiser les performances."""
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_confidence ON predictions(confidence_avg)",
            
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_endpoint ON performance_metrics(endpoint)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_status_code ON performance_metrics(status_code)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_user_id ON performance_metrics(user_id)",
            
            "CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_users_last_seen ON users(last_seen)",
            "CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active)",
            
            "CREATE INDEX IF NOT EXISTS idx_classes_name ON detected_classes(class_name)",
            "CREATE INDEX IF NOT EXISTS idx_classes_count ON detected_classes(detection_count)",
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
    
    async def save_prediction(
        self,
        user_id: str,
        image_filename: Optional[str],
        image_width: int,
        image_height: int,
        predictions: List[Dict[str, Any]],
        processing_time_ms: int,
        confidence_avg: float,
        num_detections: int
    ) -> int:
        """Sauvegarde une prédiction dans la base de données."""
        if self.pool is None:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Sauvegarder la prédiction
                prediction_id = await conn.fetchval("""
                    INSERT INTO predictions 
                    (user_id, image_filename, image_width, image_height, 
                     predictions_json, processing_time_ms, confidence_avg, num_detections)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                """, user_id, image_filename, image_width, image_height,
                    json.dumps(predictions, ensure_ascii=False),
                    processing_time_ms, confidence_avg, num_detections)
                
                # Mettre à jour les statistiques utilisateur
                await conn.execute("""
                    INSERT INTO users (user_id, total_predictions, total_detections, avg_confidence, last_seen)
                    VALUES ($1, 1, $2, $3, NOW())
                    ON CONFLICT (user_id) DO UPDATE SET
                        total_predictions = users.total_predictions + 1,
                        total_detections = users.total_detections + $2,
                        avg_confidence = (users.avg_confidence * users.total_predictions + $3) / (users.total_predictions + 1),
                        last_seen = NOW()
                """, user_id, num_detections, confidence_avg)
                
                # Mettre à jour les statistiques des classes
                for pred in predictions:
                    class_name = pred.get("class_name", "unknown")
                    await conn.execute("""
                        INSERT INTO detected_classes (class_name, detection_count, last_detected)
                        VALUES ($1, 1, NOW())
                        ON CONFLICT (class_name) DO UPDATE SET
                            detection_count = detected_classes.detection_count + 1,
                            last_detected = NOW()
                    """, class_name)
                
                return prediction_id
    
    async def get_user_history(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère l'historique des prédictions d'un utilisateur."""
        if self.pool is None:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM predictions 
                WHERE user_id = $1 
                ORDER BY timestamp DESC 
                LIMIT $2
            """, user_id, limit)
            
            return [dict(row) for row in rows]
    
    async def get_performance_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Récupère les statistiques de performance des dernières heures."""
        if self.pool is None:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as conn:
            # Statistiques générales
            general_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_requests,
                    AVG(response_time_ms) as avg_response_time,
                    MIN(response_time_ms) as min_response_time,
                    MAX(response_time_ms) as max_response_time,
                    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time
                FROM performance_metrics 
                WHERE timestamp > NOW() - INTERVAL '%s hours'
            """, hours)
            
            # Statistiques par endpoint
            endpoint_stats = await conn.fetch("""
                SELECT 
                    endpoint,
                    COUNT(*) as request_count,
                    AVG(response_time_ms) as avg_response_time,
                    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time
                FROM performance_metrics 
                WHERE timestamp > NOW() - INTERVAL '%s hours'
                GROUP BY endpoint
                ORDER BY request_count DESC
            """, hours)
            
            return {
                "general": dict(general_stats),
                "by_endpoint": [dict(row) for row in endpoint_stats],
                "period_hours": hours
            }
    
    async def get_model_usage_stats(self, days: int = 7) -> Dict[str, Any]:
        """Récupère les statistiques d'utilisation du modèle."""
        if self.pool is None:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as conn:
            # Statistiques globales
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(confidence_avg) as avg_confidence,
                    AVG(processing_time_ms) as avg_processing_time,
                    COUNT(CASE WHEN num_detections > 0 THEN 1 END) as successful_detections,
                    SUM(num_detections) as total_detections
                FROM predictions 
                WHERE timestamp > NOW() - INTERVAL '%s days'
            """, days)
            
            # Top classes détectées
            top_classes = await conn.fetch("""
                SELECT 
                    class_name,
                    detection_count
                FROM detected_classes 
                WHERE last_detected > NOW() - INTERVAL '%s days'
                ORDER BY detection_count DESC
                LIMIT 10
            """, days)
            
            return {
                **dict(stats),
                "top_classes": [dict(row) for row in top_classes],
                "period_days": days
            }
    
    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Récupère des analytics avancées pour un utilisateur."""
        if self.pool is None:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as conn:
            # Statistiques utilisateur
            user_stats = await conn.fetchrow("""
                SELECT * FROM users WHERE user_id = $1
            """, user_id)
            
            if not user_stats:
                return {"user_id": user_id, "not_found": True}
            
            # Classes préférées de l'utilisateur
            user_classes = await conn.fetch("""
                SELECT 
                    jsonb_array_elements(predictions_json)->>'class_name' as class_name,
                    COUNT(*) as count
                FROM predictions 
                WHERE user_id = $1
                GROUP BY class_name
                ORDER BY count DESC
                LIMIT 5
            """, user_id)
            
            # Activité récente
            recent_activity = await conn.fetch("""
                SELECT 
                    timestamp,
                    num_detections,
                    confidence_avg,
                    processing_time_ms
                FROM predictions 
                WHERE user_id = $1
                ORDER BY timestamp DESC
                LIMIT 10
            """, user_id)
            
            return {
                **dict(user_stats),
                "preferred_classes": [dict(row) for row in user_classes],
                "recent_activity": [dict(row) for row in recent_activity]
            }
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Nettoie les anciennes données pour maintenir la performance."""
        if self.pool is None:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Supprimer les anciennes prédictions
                deleted_predictions = await conn.fetchval("""
                    DELETE FROM predictions 
                    WHERE timestamp < NOW() - INTERVAL '%s days'
                    RETURNING COUNT(*)
                """, days_to_keep)
                
                # Supprimer les anciennes métriques
                deleted_metrics = await conn.fetchval("""
                    DELETE FROM performance_metrics 
                    WHERE timestamp < NOW() - INTERVAL '%s days'
                    RETURNING COUNT(*)
                """, days_to_keep)
                
                # VACUUM pour optimiser
                await conn.execute("VACUUM ANALYZE")
                
                logger.info(f"Cleaned up {deleted_predictions} predictions and {deleted_metrics} metrics")
    
    async def get_database_health(self) -> Dict[str, Any]:
        """Récupère l'état de santé de la base de données."""
        if self.pool is None:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as conn:
            # Taille de la base
            db_size = await conn.fetchval("SELECT pg_size_pretty(pg_database_size(current_database()))")
            
            # Nombre de connexions actives
            active_connections = await conn.fetchval("""
                SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active'
            """)
            
            # Statistiques des tables
            table_stats = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes
                FROM pg_stat_user_tables
                ORDER BY n_tup_ins DESC
            """)
            
            return {
                "database_size": db_size,
                "active_connections": active_connections,
                "table_stats": [dict(row) for row in table_stats],
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def close(self):
        """Ferme la connexion à la base de données."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")


# Instance globale de la base de données PostgreSQL
postgres_db = PostgreSQLDatabase()
