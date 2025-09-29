from __future__ import annotations

import time
import logging
from functools import wraps
from typing import Callable, Any, Optional
from contextlib import contextmanager

from .database import db

logger = logging.getLogger("plant_ai.metrics")


class PerformanceMonitor:
    """Moniteur de performance pour les endpoints API."""
    
    def __init__(self):
        self._metrics = {}
    
    @contextmanager
    def measure_time(self, operation_name: str, user_id: Optional[str] = None):
        """Contexte pour mesurer le temps d'exécution d'une opération."""
        start_time = time.time()
        error_message = None
        status_code = 200
        
        try:
            yield
        except Exception as e:
            error_message = str(e)
            status_code = 500
            raise
        finally:
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            
            # Sauvegarder la métrique
            db.save_performance_metric(
                endpoint=operation_name,
                response_time_ms=response_time_ms,
                status_code=status_code,
                user_id=user_id,
                error_message=error_message
            )
            
            # Log pour monitoring
            if status_code >= 400:
                logger.warning(f"Operation {operation_name} failed: {error_message} ({response_time_ms}ms)")
            else:
                logger.info(f"Operation {operation_name} completed in {response_time_ms}ms")
    
    def track_prediction_metrics(
        self,
        user_id: str,
        num_detections: int,
        avg_confidence: float,
        processing_time_ms: int
    ):
        """Enregistre les métriques spécifiques aux prédictions."""
        # Mettre à jour les métriques en mémoire pour un accès rapide
        if user_id not in self._metrics:
            self._metrics[user_id] = {
                "total_predictions": 0,
                "total_detections": 0,
                "avg_confidence": 0.0,
                "total_processing_time": 0
            }
        
        user_metrics = self._metrics[user_id]
        user_metrics["total_predictions"] += 1
        user_metrics["total_detections"] += num_detections
        user_metrics["total_processing_time"] += processing_time_ms
        
        # Calculer la moyenne mobile de la confiance
        if user_metrics["avg_confidence"] == 0:
            user_metrics["avg_confidence"] = avg_confidence
        else:
            user_metrics["avg_confidence"] = (
                user_metrics["avg_confidence"] * 0.9 + avg_confidence * 0.1
            )
    
    def get_user_metrics(self, user_id: str) -> dict:
        """Récupère les métriques d'un utilisateur spécifique."""
        return self._metrics.get(user_id, {
            "total_predictions": 0,
            "total_detections": 0,
            "avg_confidence": 0.0,
            "total_processing_time": 0
        })
    
    def get_global_metrics(self) -> dict:
        """Récupère les métriques globales du système."""
        total_users = len(self._metrics)
        total_predictions = sum(m["total_predictions"] for m in self._metrics.values())
        total_detections = sum(m["total_detections"] for m in self._metrics.values())
        total_processing_time = sum(m["total_processing_time"] for m in self._metrics.values())
        
        avg_confidence = 0.0
        if total_predictions > 0:
            avg_confidence = sum(
                m["avg_confidence"] * m["total_predictions"] 
                for m in self._metrics.values()
            ) / total_predictions
        
        return {
            "total_users": total_users,
            "total_predictions": total_predictions,
            "total_detections": total_detections,
            "avg_confidence": avg_confidence,
            "avg_processing_time_ms": total_processing_time / max(1, total_predictions)
        }


# Instance globale du moniteur de performance
performance_monitor = PerformanceMonitor()


def monitor_performance(endpoint_name: str):
    """Décorateur pour monitorer automatiquement les performances d'un endpoint."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extraire user_id si disponible dans les kwargs
            user_id = kwargs.get('user_id') or kwargs.get('current_user')
            
            with performance_monitor.measure_time(endpoint_name, user_id):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Extraire user_id si disponible dans les kwargs
            user_id = kwargs.get('user_id') or kwargs.get('current_user')
            
            with performance_monitor.measure_time(endpoint_name, user_id):
                return func(*args, **kwargs)
        
        # Retourner le wrapper approprié selon si la fonction est async ou non
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def calculate_prediction_metrics(predictions: list) -> dict:
    """Calcule les métriques d'une prédiction."""
    if not predictions:
        return {
            "num_detections": 0,
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "min_confidence": 0.0,
            "detected_classes": []
        }
    
    confidences = [p.get("confidence", 0.0) for p in predictions]
    classes = [p.get("class_name", "unknown") for p in predictions]
    
    return {
        "num_detections": len(predictions),
        "avg_confidence": sum(confidences) / len(confidences),
        "max_confidence": max(confidences),
        "min_confidence": min(confidences),
        "detected_classes": list(set(classes))
    }

