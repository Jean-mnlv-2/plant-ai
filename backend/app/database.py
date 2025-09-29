from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .settings import settings


class Database:
    """Gestionnaire de base de données SQLite pour Plant-AI."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialise les tables de la base de données."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        image_filename TEXT,
                        image_width INTEGER,
                        image_height INTEGER,
                        predictions_json TEXT NOT NULL,
                        processing_time_ms INTEGER,
                        confidence_avg REAL,
                        num_detections INTEGER
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        endpoint TEXT NOT NULL,
                        response_time_ms INTEGER NOT NULL,
                        status_code INTEGER NOT NULL,
                        user_id TEXT,
                        error_message TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_usage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        model_version TEXT NOT NULL,
                        predictions_count INTEGER NOT NULL,
                        avg_confidence REAL,
                        unique_users INTEGER
                    )
                """)
                
                # Index pour améliorer les performances
                conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_endpoint ON performance_metrics(endpoint)")
                
                conn.commit()
    
    def save_prediction(
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
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    INSERT INTO predictions 
                    (user_id, timestamp, image_filename, image_width, image_height, 
                     predictions_json, processing_time_ms, confidence_avg, num_detections)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    datetime.utcnow().isoformat(),
                    image_filename,
                    image_width,
                    image_height,
                    json.dumps(predictions, ensure_ascii=False),
                    processing_time_ms,
                    confidence_avg,
                    num_detections
                ))
                conn.commit()
                return cursor.lastrowid or 0
    
    def get_user_history(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère l'historique des prédictions d'un utilisateur."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM predictions 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (user_id, limit))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
    
    def get_all_history(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Récupère l'historique global des prédictions."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM predictions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
    
    def save_performance_metric(
        self,
        endpoint: str,
        response_time_ms: int,
        status_code: int,
        user_id: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """Sauvegarde une métrique de performance."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (timestamp, endpoint, response_time_ms, status_code, user_id, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.utcnow().isoformat(),
                    endpoint,
                    response_time_ms,
                    status_code,
                    user_id,
                    error_message
                ))
                conn.commit()
    
    def get_performance_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Récupère les statistiques de performance des dernières heures."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                
                # Statistiques générales
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_requests,
                        AVG(response_time_ms) as avg_response_time,
                        MIN(response_time_ms) as min_response_time,
                        MAX(response_time_ms) as max_response_time,
                        COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count
                    FROM performance_metrics 
                    WHERE timestamp > datetime('now', '-{} hours')
                """.format(hours))
                
                general_stats = dict(cursor.fetchone())
                
                # Statistiques par endpoint
                cursor = conn.execute("""
                    SELECT 
                        endpoint,
                        COUNT(*) as request_count,
                        AVG(response_time_ms) as avg_response_time,
                        COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count
                    FROM performance_metrics 
                    WHERE timestamp > datetime('now', '-{} hours')
                    GROUP BY endpoint
                    ORDER BY request_count DESC
                """.format(hours))
                
                endpoint_stats = [dict(row) for row in cursor.fetchall()]
                
                return {
                    "general": general_stats,
                    "by_endpoint": endpoint_stats,
                    "period_hours": hours
                }
    
    def get_model_usage_stats(self, days: int = 7) -> Dict[str, Any]:
        """Récupère les statistiques d'utilisation du modèle."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                
                # Statistiques globales
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        COUNT(DISTINCT user_id) as unique_users,
                        AVG(confidence_avg) as avg_confidence,
                        AVG(processing_time_ms) as avg_processing_time,
                        COUNT(CASE WHEN num_detections > 0 THEN 1 END) as successful_detections
                    FROM predictions 
                    WHERE timestamp > datetime('now', '-{} days')
                """.format(days))
                
                stats = dict(cursor.fetchone())
                
                # Top classes détectées
                cursor = conn.execute("""
                    SELECT 
                        json_extract(predictions_json, '$[0].class_name') as class_name,
                        COUNT(*) as detection_count
                    FROM predictions 
                    WHERE timestamp > datetime('now', '-{} days')
                    AND json_extract(predictions_json, '$[0].class_name') IS NOT NULL
                    GROUP BY class_name
                    ORDER BY detection_count DESC
                    LIMIT 10
                """.format(days))
                
                top_classes = [dict(row) for row in cursor.fetchall()]
                
                return {
                    **stats,
                    "top_classes": top_classes,
                    "period_days": days
                }
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Nettoie les anciennes données pour maintenir la performance."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Garder seulement les données des derniers jours
                cutoff_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                from datetime import timedelta
                cutoff_date = cutoff_date - timedelta(days=days_to_keep)
                
                # Supprimer les anciennes prédictions
                conn.execute("""
                    DELETE FROM predictions 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                # Supprimer les anciennes métriques
                conn.execute("""
                    DELETE FROM performance_metrics 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                conn.commit()
                
                # VACUUM pour optimiser la base
                conn.execute("VACUUM")


# Instance globale de la base de données
db = Database()
