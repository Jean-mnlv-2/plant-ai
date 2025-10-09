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

                # Tables pour feedback et cas incertains
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        timestamp DATETIME NOT NULL,
                        image_filename TEXT,
                        original_class TEXT,
                        corrected_class TEXT,
                        notes TEXT,
                        prediction_id INTEGER
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS uncertain_cases (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        timestamp DATETIME NOT NULL,
                        image_filename TEXT,
                        image_width INTEGER,
                        image_height INTEGER,
                        predictions_json TEXT,
                        reason TEXT,
                        min_confidence REAL,
                        max_confidence REAL
                    )
                """)
                
                # Table des utilisateurs
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        country TEXT NOT NULL,
                        role TEXT NOT NULL DEFAULT 'farmer',
                        preferences TEXT,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    )
                """)
                
                # Table des diagnostics
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS diagnostics (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        images TEXT NOT NULL,
                        results TEXT NOT NULL,
                        location TEXT,
                        plant_type TEXT,
                        status TEXT DEFAULT 'completed',
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
    
    def save_prediction(
        self,
        user_id: str,
        image_filename: Optional[str],
        image_width: int,
        image_height: int,
        predictions: List[Dict[str, Any]],
        processing_time_ms: int,
        confidence_avg: Optional[float] = None,
        num_detections: Optional[int] = None
    ) -> int:
        """Sauvegarde une prédiction dans la base de données."""
        # Calculer les valeurs par défaut si non fournies
        if confidence_avg is None:
            confidence_avg = sum(p.get('confidence', 0) for p in predictions) / len(predictions) if predictions else 0.0
        
        if num_detections is None:
            num_detections = len(predictions)
        
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

    def save_uncertain_case(
        self,
        user_id: Optional[str],
        image_filename: Optional[str],
        image_width: int,
        image_height: int,
        predictions: List[Dict[str, Any]],
        reason: str,
        min_confidence: float,
        max_confidence: float
    ) -> int:
        """Enregistre un cas incertain pour annotation ultérieure."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    INSERT INTO uncertain_cases 
                    (user_id, timestamp, image_filename, image_width, image_height, predictions_json, reason, min_confidence, max_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    datetime.utcnow().isoformat(),
                    image_filename,
                    image_width,
                    image_height,
                    json.dumps(predictions, ensure_ascii=False),
                    reason,
                    min_confidence,
                    max_confidence
                ))
                conn.commit()
                return cursor.lastrowid or 0

    def list_uncertain_cases(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM uncertain_cases 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    def save_feedback(
        self,
        user_id: Optional[str],
        image_filename: Optional[str],
        original_class: Optional[str],
        corrected_class: str,
        notes: Optional[str],
        prediction_id: Optional[int]
    ) -> int:
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    INSERT INTO feedback 
                    (user_id, timestamp, image_filename, original_class, corrected_class, notes, prediction_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    datetime.utcnow().isoformat(),
                    image_filename,
                    original_class,
                    corrected_class,
                    notes,
                    prediction_id
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

    def list_users(self) -> List[str]:
        """Liste des utilisateurs ayant des prédictions enregistrées."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT DISTINCT user_id FROM predictions WHERE user_id IS NOT NULL ORDER BY user_id
                """)
                rows = cursor.fetchall()
                return [row["user_id"] for row in rows]
    
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
    
    def create_user(self, user_data: Dict[str, Any], password_hash: str):
        """Créer un nouvel utilisateur."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT INTO users (id, name, email, password_hash, country, role, preferences, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_data["id"],
                    user_data["name"],
                    user_data["email"],
                    password_hash,
                    user_data["country"],
                    user_data["role"],
                    json.dumps(user_data.get("preferences", {})),
                    user_data["createdAt"],
                    datetime.utcnow()
                ))
                conn.commit()
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Récupérer un utilisateur par email."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT id, name, email, country, role, preferences, created_at
                    FROM users WHERE email = ?
                """, (email,))
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "name": row[1],
                        "email": row[2],
                        "country": row[3],
                        "role": row[4],
                        "preferences": json.loads(row[5]) if row[5] else {},
                        "createdAt": row[6]
                    }
                return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Récupérer un utilisateur par ID."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT id, name, email, country, role, preferences, created_at
                    FROM users WHERE id = ?
                """, (user_id,))
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "name": row[1],
                        "email": row[2],
                        "country": row[3],
                        "role": row[4],
                        "preferences": json.loads(row[5]) if row[5] else {},
                        "createdAt": row[6]
                    }
                return None
    
    def get_user_password(self, email: str) -> Optional[str]:
        """Récupérer le hash du mot de passe d'un utilisateur."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT password_hash FROM users WHERE email = ?
                """, (email,))
                row = cursor.fetchone()
                return row[0] if row else None
    
    def save_diagnostic(self, diagnostic_data: Dict[str, Any]):
        """Sauvegarder un diagnostic."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT INTO diagnostics (id, user_id, images, results, location, plant_type, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    diagnostic_data["id"],
                    diagnostic_data["userId"],
                    json.dumps(diagnostic_data["images"]),
                    json.dumps(diagnostic_data["results"]),
                    json.dumps(diagnostic_data.get("location", {})),
                    diagnostic_data.get("plantType", ""),
                    diagnostic_data.get("status", "completed"),
                    diagnostic_data["createdAt"],
                    datetime.utcnow()
                ))
                conn.commit()
    
    def get_diagnostics(self, user_id: str, limit: int = 10, offset: int = 0, 
                       sort_by: str = "created_at", order: str = "desc") -> Dict[str, Any]:
        """Récupérer les diagnostics d'un utilisateur."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Compter le total
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM diagnostics WHERE user_id = ?
                """, (user_id,))
                total = cursor.fetchone()[0]
                
                # Récupérer les diagnostics
                order_clause = f"ORDER BY {sort_by} {order.upper()}"
                cursor = conn.execute(f"""
                    SELECT id, user_id, images, results, location, plant_type, status, created_at, updated_at
                    FROM diagnostics WHERE user_id = ?
                    {order_clause}
                    LIMIT ? OFFSET ?
                """, (user_id, limit, offset))
                
                diagnostics = []
                for row in cursor.fetchall():
                    diagnostics.append({
                        "id": row[0],
                        "userId": row[1],
                        "images": json.loads(row[2]),
                        "results": json.loads(row[3]),
                        "location": json.loads(row[4]) if row[4] else {},
                        "plantType": row[5],
                        "status": row[6],
                        "createdAt": row[7],
                        "updatedAt": row[8]
                    })
                
                return {
                    "success": True,
                    "diagnostics": diagnostics,
                    "pagination": {
                        "total": total,
                        "limit": limit,
                        "offset": offset,
                        "hasMore": offset + limit < total
                    }
                }
    
    def get_diagnostic_by_id(self, diagnostic_id: str) -> Optional[Dict[str, Any]]:
        """Récupérer un diagnostic par ID."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT id, user_id, images, results, location, plant_type, status, created_at, updated_at
                    FROM diagnostics WHERE id = ?
                """, (diagnostic_id,))
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "userId": row[1],
                        "images": json.loads(row[2]),
                        "results": json.loads(row[3]),
                        "location": json.loads(row[4]) if row[4] else {},
                        "plantType": row[5],
                        "status": row[6],
                        "createdAt": row[7],
                        "updatedAt": row[8]
                    }
                return None
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Récupérer les statistiques d'un utilisateur."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Statistiques des prédictions
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        AVG(confidence_avg) as avg_confidence,
                        COUNT(DISTINCT DATE(timestamp)) as active_days
                    FROM predictions 
                    WHERE user_id = ?
                """, (user_id,))
                pred_stats = cursor.fetchone()
                
                # Statistiques des diagnostics
                cursor = conn.execute("""
                    SELECT COUNT(*) as total_diagnostics
                    FROM diagnostics 
                    WHERE user_id = ?
                """, (user_id,))
                diag_stats = cursor.fetchone()
                
                return {
                    "totalPredictions": pred_stats[0] or 0,
                    "averageConfidence": round(pred_stats[1] or 0, 2),
                    "activeDays": pred_stats[2] or 0,
                    "totalDiagnostics": diag_stats[0] or 0
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
