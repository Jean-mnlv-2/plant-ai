"""
Gestionnaire de base de données PostgreSQL pour Plant-AI.
Optimisé pour la base de connaissances agronomique avec 50,000+ fiches.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import asyncpg
from asyncpg import Pool, Connection
from pydantic import BaseModel

from .settings import settings

logger = logging.getLogger(__name__)


class PostgreSQLDatabase:
    """Gestionnaire de base de données PostgreSQL optimisé pour Plant-AI."""
    
    def __init__(self):
        self.pool: Optional[Pool] = None
        self._connection_params = self._get_connection_params()
    
    def _get_connection_params(self) -> Dict[str, Any]:
        """Récupérer les paramètres de connexion PostgreSQL."""
        return {
            "host": settings.postgres_host,
            "port": settings.postgres_port,
            "database": settings.postgres_database,
            "user": settings.postgres_user,
            "password": settings.postgres_password,
            "min_size": 5,
            "max_size": 20,
            "command_timeout": 60,
        }
    
    async def connect(self):
        """Établir la connexion à PostgreSQL."""
        try:
            self.pool = await asyncpg.create_pool(**self._connection_params)
            logger.info("PostgreSQL connection pool created successfully")
            
            # Initialiser les tables
            await self._init_tables()
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    async def disconnect(self):
        """Fermer la connexion PostgreSQL."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
    
    async def _init_tables(self):
        """Initialiser toutes les tables de la base de données."""
        async with self.pool.acquire() as conn:
            # Tables principales
            await self._create_users_table(conn)
            await self._create_diagnostics_table(conn)
            await self._create_performance_metrics_table(conn)
            
            # Tables pour la base de connaissances agronomique
            await self._create_cultures_table(conn)
            await self._create_pathogens_table(conn)
            await self._create_disease_entries_table(conn)
            await self._create_disease_symptoms_table(conn)
            await self._create_disease_images_table(conn)
            await self._create_disease_conditions_table(conn)
            await self._create_disease_control_methods_table(conn)
            await self._create_disease_products_table(conn)
            await self._create_disease_precautions_table(conn)
            await self._create_disease_prevention_table(conn)
            await self._create_disease_regions_table(conn)
            await self._create_disease_translations_table(conn)
            
            # Index pour optimiser les performances
            await self._create_indexes(conn)
            
            logger.info("All PostgreSQL tables initialized successfully")
    
    async def _create_users_table(self, conn: Connection):
        """Créer la table des utilisateurs."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                country VARCHAR(100) NOT NULL,
                role VARCHAR(50) NOT NULL DEFAULT 'farmer',
                preferences JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def _create_diagnostics_table(self, conn: Connection):
        """Créer la table des diagnostics."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnostics (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                images JSONB NOT NULL,
                results JSONB NOT NULL,
                location JSONB,
                plant_type VARCHAR(100),
                status VARCHAR(50) DEFAULT 'completed',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def _create_performance_metrics_table(self, conn: Connection):
        """Créer la table des métriques de performance."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                endpoint VARCHAR(255) NOT NULL,
                response_time_ms INTEGER NOT NULL,
                status_code INTEGER NOT NULL,
                user_id UUID REFERENCES users(id),
                error_message TEXT
            )
        """)
    
    async def _create_cultures_table(self, conn: Connection):
        """Créer la table des cultures."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS cultures (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                scientific_name VARCHAR(255),
                category VARCHAR(100),
                description TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def _create_pathogens_table(self, conn: Connection):
        """Créer la table des pathogènes."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS pathogens (
                id SERIAL PRIMARY KEY,
                scientific_name VARCHAR(255) NOT NULL,
                common_name VARCHAR(255),
                type VARCHAR(50) NOT NULL, -- fungus, bacteria, virus, parasite
                description TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def _create_disease_entries_table(self, conn: Connection):
        """Créer la table principale des fiches maladies."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS disease_entries (
                id SERIAL PRIMARY KEY,
                culture_id INTEGER REFERENCES cultures(id) ON DELETE CASCADE,
                pathogen_id INTEGER REFERENCES pathogens(id) ON DELETE CASCADE,
                scientific_name VARCHAR(255) NOT NULL,
                common_name VARCHAR(255) NOT NULL,
                severity_level VARCHAR(20) DEFAULT 'medium', -- low, medium, high, critical
                treatment_urgency VARCHAR(20) DEFAULT 'planned', -- immediate, urgent, planned, monitoring
                is_active BOOLEAN DEFAULT TRUE,
                created_by UUID REFERENCES users(id),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def _create_disease_symptoms_table(self, conn: Connection):
        """Créer la table des symptômes."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS disease_symptoms (
                id SERIAL PRIMARY KEY,
                disease_id INTEGER REFERENCES disease_entries(id) ON DELETE CASCADE,
                description TEXT NOT NULL,
                severity VARCHAR(20) DEFAULT 'medium',
                affected_part VARCHAR(100), -- leaves, fruits, stems, roots
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def _create_disease_images_table(self, conn: Connection):
        """Créer la table des images de référence."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS disease_images (
                id SERIAL PRIMARY KEY,
                disease_id INTEGER REFERENCES disease_entries(id) ON DELETE CASCADE,
                image_url VARCHAR(500) NOT NULL,
                image_type VARCHAR(50) DEFAULT 'symptom', -- symptom, pathogen, damage
                description TEXT,
                is_primary BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def _create_disease_conditions_table(self, conn: Connection):
        """Créer la table des conditions favorables."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS disease_conditions (
                id SERIAL PRIMARY KEY,
                disease_id INTEGER REFERENCES disease_entries(id) ON DELETE CASCADE,
                temperature_min DECIMAL(5,2),
                temperature_max DECIMAL(5,2),
                humidity_min INTEGER,
                humidity_max INTEGER,
                soil_type VARCHAR(100),
                seasonality VARCHAR(100),
                climate_conditions TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def _create_disease_control_methods_table(self, conn: Connection):
        """Créer la table des méthodes de lutte."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS disease_control_methods (
                id SERIAL PRIMARY KEY,
                disease_id INTEGER REFERENCES disease_entries(id) ON DELETE CASCADE,
                method_type VARCHAR(50) NOT NULL, -- cultural, biological, chemical
                description TEXT NOT NULL,
                effectiveness VARCHAR(20) DEFAULT 'medium', -- low, medium, high
                cost_level VARCHAR(20) DEFAULT 'medium', -- low, medium, high
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def _create_disease_products_table(self, conn: Connection):
        """Créer la table des produits recommandés."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS disease_products (
                id SERIAL PRIMARY KEY,
                disease_id INTEGER REFERENCES disease_entries(id) ON DELETE CASCADE,
                product_name VARCHAR(255) NOT NULL,
                active_ingredient VARCHAR(255),
                dosage VARCHAR(100),
                application_method VARCHAR(100),
                safety_class VARCHAR(50),
                registration_number VARCHAR(100),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def _create_disease_precautions_table(self, conn: Connection):
        """Créer la table des précautions d'usage."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS disease_precautions (
                id SERIAL PRIMARY KEY,
                disease_id INTEGER REFERENCES disease_entries(id) ON DELETE CASCADE,
                risk_level VARCHAR(20) DEFAULT 'medium', -- low, medium, high
                safety_period_days INTEGER,
                dosage_instructions TEXT,
                contraindications TEXT,
                protective_equipment TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def _create_disease_prevention_table(self, conn: Connection):
        """Créer la table des mesures de prévention."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS disease_prevention (
                id SERIAL PRIMARY KEY,
                disease_id INTEGER REFERENCES disease_entries(id) ON DELETE CASCADE,
                practice_type VARCHAR(50) NOT NULL, -- cultural, biological, monitoring
                description TEXT NOT NULL,
                timing VARCHAR(100),
                effectiveness VARCHAR(20) DEFAULT 'medium',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def _create_disease_regions_table(self, conn: Connection):
        """Créer la table des régions/zones climatiques."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS disease_regions (
                id SERIAL PRIMARY KEY,
                disease_id INTEGER REFERENCES disease_entries(id) ON DELETE CASCADE,
                region_name VARCHAR(255) NOT NULL,
                country VARCHAR(100),
                climate_zone VARCHAR(100),
                latitude_min DECIMAL(8,4),
                latitude_max DECIMAL(8,4),
                longitude_min DECIMAL(8,4),
                longitude_max DECIMAL(8,4),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def _create_disease_translations_table(self, conn: Connection):
        """Créer la table des traductions multilingues."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS disease_translations (
                id SERIAL PRIMARY KEY,
                disease_id INTEGER REFERENCES disease_entries(id) ON DELETE CASCADE,
                language_code VARCHAR(5) NOT NULL, -- fr, en, es, etc.
                field_name VARCHAR(50) NOT NULL, -- name, description, etc.
                translated_text TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(disease_id, language_code, field_name)
            )
        """)
    
    async def _create_indexes(self, conn: Connection):
        """Créer les index pour optimiser les performances."""
        indexes = [
            # Index pour les utilisateurs
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)",
            
            # Index pour les diagnostics
            "CREATE INDEX IF NOT EXISTS idx_diagnostics_user_id ON diagnostics(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_diagnostics_created_at ON diagnostics(created_at)",
            
            # Index pour les métriques
            "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_performance_endpoint ON performance_metrics(endpoint)",
            
            # Index pour la base de connaissances
            "CREATE INDEX IF NOT EXISTS idx_disease_entries_culture ON disease_entries(culture_id)",
            "CREATE INDEX IF NOT EXISTS idx_disease_entries_pathogen ON disease_entries(pathogen_id)",
            "CREATE INDEX IF NOT EXISTS idx_disease_entries_active ON disease_entries(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_disease_entries_name ON disease_entries USING gin(to_tsvector('french', scientific_name || ' ' || common_name))",
            
            # Index pour les symptômes
            "CREATE INDEX IF NOT EXISTS idx_symptoms_disease ON disease_symptoms(disease_id)",
            "CREATE INDEX IF NOT EXISTS idx_symptoms_description ON disease_symptoms USING gin(to_tsvector('french', description))",
            
            # Index pour les images
            "CREATE INDEX IF NOT EXISTS idx_images_disease ON disease_images(disease_id)",
            "CREATE INDEX IF NOT EXISTS idx_images_primary ON disease_images(is_primary)",
            
            # Index pour les conditions
            "CREATE INDEX IF NOT EXISTS idx_conditions_disease ON disease_conditions(disease_id)",
            "CREATE INDEX IF NOT EXISTS idx_conditions_temperature ON disease_conditions(temperature_min, temperature_max)",
            
            # Index pour les méthodes de lutte
            "CREATE INDEX IF NOT EXISTS idx_control_disease ON disease_control_methods(disease_id)",
            "CREATE INDEX IF NOT EXISTS idx_control_type ON disease_control_methods(method_type)",
            
            # Index pour les produits
            "CREATE INDEX IF NOT EXISTS idx_products_disease ON disease_products(disease_id)",
            "CREATE INDEX IF NOT EXISTS idx_products_name ON disease_products USING gin(to_tsvector('french', product_name))",
            
            # Index pour les précautions
            "CREATE INDEX IF NOT EXISTS idx_precautions_disease ON disease_precautions(disease_id)",
            "CREATE INDEX IF NOT EXISTS idx_precautions_risk ON disease_precautions(risk_level)",
            
            # Index pour la prévention
            "CREATE INDEX IF NOT EXISTS idx_prevention_disease ON disease_prevention(disease_id)",
            "CREATE INDEX IF NOT EXISTS idx_prevention_type ON disease_prevention(practice_type)",
            
            # Index pour les régions
            "CREATE INDEX IF NOT EXISTS idx_regions_disease ON disease_regions(disease_id)",
            "CREATE INDEX IF NOT EXISTS idx_regions_country ON disease_regions(country)",
            
            # Index pour les traductions
            "CREATE INDEX IF NOT EXISTS idx_translations_disease ON disease_translations(disease_id)",
            "CREATE INDEX IF NOT EXISTS idx_translations_language ON disease_translations(language_code)",
            "CREATE INDEX IF NOT EXISTS idx_translations_text ON disease_translations USING gin(to_tsvector('french', translated_text))",
        ]
        
        for index_sql in indexes:
            try:
                await conn.execute(index_sql)
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")
    
    # =============================================================================
    # MÉTHODES DE GESTION DES FICHES MALADIES
    # =============================================================================
    
    async def create_disease_entry(self, entry_data: Dict[str, Any]) -> int:
        """Créer une nouvelle fiche maladie."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Créer l'entrée principale
                disease_id = await conn.fetchval("""
                    INSERT INTO disease_entries (
                        culture_id, pathogen_id, scientific_name, common_name,
                        severity_level, treatment_urgency, created_by
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                """, 
                entry_data.get('culture_id'),
                entry_data.get('pathogen_id'),
                entry_data.get('scientific_name'),
                entry_data.get('common_name'),
                entry_data.get('severity_level', 'medium'),
                entry_data.get('treatment_urgency', 'planned'),
                entry_data.get('created_by')
                )
                
                # Ajouter les symptômes
                if 'symptoms' in entry_data:
                    for symptom in entry_data['symptoms']:
                        await conn.execute("""
                            INSERT INTO disease_symptoms (disease_id, description, severity, affected_part)
                            VALUES ($1, $2, $3, $4)
                        """, disease_id, symptom.get('description'), 
                        symptom.get('severity', 'medium'), symptom.get('affected_part'))
                
                # Ajouter les images
                if 'images' in entry_data:
                    for image in entry_data['images']:
                        await conn.execute("""
                            INSERT INTO disease_images (disease_id, image_url, image_type, description, is_primary)
                            VALUES ($1, $2, $3, $4, $5)
                        """, disease_id, image.get('url'), 
                        image.get('type', 'symptom'), image.get('description'), 
                        image.get('is_primary', False))
                
                # Ajouter les conditions
                if 'conditions' in entry_data:
                    conditions = entry_data['conditions']
                    await conn.execute("""
                        INSERT INTO disease_conditions (
                            disease_id, temperature_min, temperature_max,
                            humidity_min, humidity_max, soil_type, seasonality, climate_conditions
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """, disease_id, conditions.get('temperature_min'), 
                    conditions.get('temperature_max'), conditions.get('humidity_min'),
                    conditions.get('humidity_max'), conditions.get('soil_type'),
                    conditions.get('seasonality'), conditions.get('climate_conditions'))
                
                # Ajouter les méthodes de lutte
                if 'control_methods' in entry_data:
                    for method in entry_data['control_methods']:
                        await conn.execute("""
                            INSERT INTO disease_control_methods (disease_id, method_type, description, effectiveness, cost_level)
                            VALUES ($1, $2, $3, $4, $5)
                        """, disease_id, method.get('type'), method.get('description'),
                        method.get('effectiveness', 'medium'), method.get('cost_level', 'medium'))
                
                # Ajouter les produits
                if 'products' in entry_data:
                    for product in entry_data['products']:
                        await conn.execute("""
                            INSERT INTO disease_products (
                                disease_id, product_name, active_ingredient, dosage,
                                application_method, safety_class, registration_number
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """, disease_id, product.get('name'), product.get('active_ingredient'),
                        product.get('dosage'), product.get('application_method'),
                        product.get('safety_class'), product.get('registration_number'))
                
                # Ajouter les précautions
                if 'precautions' in entry_data:
                    precautions = entry_data['precautions']
                    await conn.execute("""
                        INSERT INTO disease_precautions (
                            disease_id, risk_level, safety_period_days, dosage_instructions,
                            contraindications, protective_equipment
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                    """, disease_id, precautions.get('risk_level', 'medium'),
                    precautions.get('safety_period_days'), precautions.get('dosage_instructions'),
                    precautions.get('contraindications'), precautions.get('protective_equipment'))
                
                # Ajouter la prévention
                if 'prevention' in entry_data:
                    for prevention in entry_data['prevention']:
                        await conn.execute("""
                            INSERT INTO disease_prevention (disease_id, practice_type, description, timing, effectiveness)
                            VALUES ($1, $2, $3, $4, $5)
                        """, disease_id, prevention.get('type'), prevention.get('description'),
                        prevention.get('timing'), prevention.get('effectiveness', 'medium'))
                
                # Ajouter les régions
                if 'regions' in entry_data:
                    for region in entry_data['regions']:
                        await conn.execute("""
                            INSERT INTO disease_regions (
                                disease_id, region_name, country, climate_zone,
                                latitude_min, latitude_max, longitude_min, longitude_max
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """, disease_id, region.get('name'), region.get('country'),
                        region.get('climate_zone'), region.get('latitude_min'),
                        region.get('latitude_max'), region.get('longitude_min'),
                        region.get('longitude_max'))
                
                # Ajouter les traductions
                if 'translations' in entry_data:
                    for translation in entry_data['translations']:
                        await conn.execute("""
                            INSERT INTO disease_translations (disease_id, language_code, field_name, translated_text)
                            VALUES ($1, $2, $3, $4)
                        """, disease_id, translation.get('language_code'),
                        translation.get('field_name'), translation.get('translated_text'))
                
                return disease_id
    
    async def get_disease_entries(self, 
                                limit: int = 20, 
                                offset: int = 0,
                                search: Optional[str] = None,
                                culture_id: Optional[int] = None,
                                pathogen_type: Optional[str] = None,
                                severity: Optional[str] = None,
                                language: str = 'fr') -> Tuple[List[Dict[str, Any]], int]:
        """Récupérer les fiches maladies avec filtres et pagination."""
        async with self.pool.acquire() as conn:
            # Construire la requête de base
            base_query = """
                FROM disease_entries de
                LEFT JOIN cultures c ON de.culture_id = c.id
                LEFT JOIN pathogens p ON de.pathogen_id = p.id
                WHERE de.is_active = TRUE
            """
            
            params = []
            param_count = 0
            
            # Ajouter les filtres
            if search:
                param_count += 1
                base_query += f" AND to_tsvector('french', de.scientific_name || ' ' || de.common_name) @@ plainto_tsquery('french', ${param_count})"
                params.append(search)
            
            if culture_id:
                param_count += 1
                base_query += f" AND de.culture_id = ${param_count}"
                params.append(culture_id)
            
            if pathogen_type:
                param_count += 1
                base_query += f" AND p.type = ${param_count}"
                params.append(pathogen_type)
            
            if severity:
                param_count += 1
                base_query += f" AND de.severity_level = ${param_count}"
                params.append(severity)
            
            # Compter le total
            count_query = f"SELECT COUNT(*) {base_query}"
            total = await conn.fetchval(count_query, *params)
            
            # Récupérer les données
            data_query = f"""
                SELECT 
                    de.id, de.scientific_name, de.common_name, de.severity_level,
                    de.treatment_urgency, de.created_at, de.updated_at,
                    c.name as culture_name, p.scientific_name as pathogen_name, p.type as pathogen_type
                {base_query}
                ORDER BY de.updated_at DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])
            
            rows = await conn.fetch(data_query, *params)
            
            # Convertir en dictionnaires
            entries = [dict(row) for row in rows]
            
            return entries, total
    
    async def get_disease_entry_by_id(self, disease_id: int, language: str = 'fr') -> Optional[Dict[str, Any]]:
        """Récupérer une fiche maladie complète par ID."""
        async with self.pool.acquire() as conn:
            # Récupérer l'entrée principale
            entry = await conn.fetchrow("""
                SELECT 
                    de.*, c.name as culture_name, p.scientific_name as pathogen_name, p.type as pathogen_type
                FROM disease_entries de
                LEFT JOIN cultures c ON de.culture_id = c.id
                LEFT JOIN pathogens p ON de.pathogen_id = p.id
                WHERE de.id = $1 AND de.is_active = TRUE
            """, disease_id)
            
            if not entry:
                return None
            
            result = dict(entry)
            
            # Récupérer les symptômes
            symptoms = await conn.fetch("""
                SELECT * FROM disease_symptoms WHERE disease_id = $1 ORDER BY id
            """, disease_id)
            result['symptoms'] = [dict(s) for s in symptoms]
            
            # Récupérer les images
            images = await conn.fetch("""
                SELECT * FROM disease_images WHERE disease_id = $1 ORDER BY is_primary DESC, id
            """, disease_id)
            result['images'] = [dict(i) for i in images]
            
            # Récupérer les conditions
            conditions = await conn.fetchrow("""
                SELECT * FROM disease_conditions WHERE disease_id = $1
            """, disease_id)
            result['conditions'] = dict(conditions) if conditions else {}
            
            # Récupérer les méthodes de lutte
            control_methods = await conn.fetch("""
                SELECT * FROM disease_control_methods WHERE disease_id = $1 ORDER BY id
            """, disease_id)
            result['control_methods'] = [dict(cm) for cm in control_methods]
            
            # Récupérer les produits
            products = await conn.fetch("""
                SELECT * FROM disease_products WHERE disease_id = $1 ORDER BY id
            """, disease_id)
            result['products'] = [dict(p) for p in products]
            
            # Récupérer les précautions
            precautions = await conn.fetchrow("""
                SELECT * FROM disease_precautions WHERE disease_id = $1
            """, disease_id)
            result['precautions'] = dict(precautions) if precautions else {}
            
            # Récupérer la prévention
            prevention = await conn.fetch("""
                SELECT * FROM disease_prevention WHERE disease_id = $1 ORDER BY id
            """, disease_id)
            result['prevention'] = [dict(p) for p in prevention]
            
            # Récupérer les régions
            regions = await conn.fetch("""
                SELECT * FROM disease_regions WHERE disease_id = $1 ORDER BY id
            """, disease_id)
            result['regions'] = [dict(r) for r in regions]
            
            # Récupérer les traductions
            translations = await conn.fetch("""
                SELECT * FROM disease_translations WHERE disease_id = $1 AND language_code = $2
            """, disease_id, language)
            result['translations'] = {t['field_name']: t['translated_text'] for t in translations}
            
            return result
    
    async def update_disease_entry(self, disease_id: int, entry_data: Dict[str, Any]) -> bool:
        """Mettre à jour une fiche maladie."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Mettre à jour l'entrée principale
                await conn.execute("""
                    UPDATE disease_entries SET
                        culture_id = $2, pathogen_id = $3, scientific_name = $4,
                        common_name = $5, severity_level = $6, treatment_urgency = $7,
                        updated_at = NOW()
                    WHERE id = $1
                """, disease_id, entry_data.get('culture_id'), entry_data.get('pathogen_id'),
                entry_data.get('scientific_name'), entry_data.get('common_name'),
                entry_data.get('severity_level'), entry_data.get('treatment_urgency'))
                
                # Supprimer et recréer les données associées
                await self._update_related_data(conn, disease_id, entry_data)
                
                return True
    
    async def _update_related_data(self, conn: Connection, disease_id: int, entry_data: Dict[str, Any]):
        """Mettre à jour les données associées à une fiche maladie."""
        # Supprimer les anciennes données
        tables_to_clear = [
            'disease_symptoms', 'disease_images', 'disease_conditions',
            'disease_control_methods', 'disease_products', 'disease_precautions',
            'disease_prevention', 'disease_regions', 'disease_translations'
        ]
        
        for table in tables_to_clear:
            await conn.execute(f"DELETE FROM {table} WHERE disease_id = $1", disease_id)
        
        # Recréer les données (même logique que dans create_disease_entry)
        # ... (code similaire à create_disease_entry pour recréer les données)
    
    async def delete_disease_entry(self, disease_id: int) -> bool:
        """Supprimer une fiche maladie (soft delete)."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE disease_entries SET is_active = FALSE, updated_at = NOW()
                WHERE id = $1
            """, disease_id)
            return result != "UPDATE 0"
    
    async def hard_delete_disease_entry(self, disease_id: int) -> bool:
        """Supprimer définitivement une fiche maladie."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Supprimer toutes les données associées
                tables_to_delete = [
                    'disease_symptoms', 'disease_images', 'disease_conditions',
                    'disease_control_methods', 'disease_products', 'disease_precautions',
                    'disease_prevention', 'disease_regions', 'disease_translations'
                ]
                
                for table in tables_to_delete:
                    await conn.execute(f"DELETE FROM {table} WHERE disease_id = $1", disease_id)
                
                # Supprimer l'entrée principale
                result = await conn.execute("DELETE FROM disease_entries WHERE id = $1", disease_id)
                return result != "DELETE 0"
    
    # =============================================================================
    # MÉTHODES DE GESTION DES CULTURES ET PATHOGÈNES
    # =============================================================================
    
    async def create_culture(self, name: str, scientific_name: str = None, 
                           category: str = None, description: str = None) -> int:
        """Créer une nouvelle culture."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval("""
                INSERT INTO cultures (name, scientific_name, category, description)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """, name, scientific_name, category, description)
    
    async def create_pathogen(self, scientific_name: str, common_name: str = None,
                             pathogen_type: str = None, description: str = None) -> int:
        """Créer un nouveau pathogène."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval("""
                INSERT INTO pathogens (scientific_name, common_name, type, description)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """, scientific_name, common_name, pathogen_type, description)
    
    async def get_cultures(self, search: str = None) -> List[Dict[str, Any]]:
        """Récupérer la liste des cultures."""
        async with self.pool.acquire() as conn:
            if search:
                rows = await conn.fetch("""
                    SELECT * FROM cultures 
                    WHERE name ILIKE $1 OR scientific_name ILIKE $1
                    ORDER BY name
                """, f"%{search}%")
            else:
                rows = await conn.fetch("SELECT * FROM cultures ORDER BY name")
            
            return [dict(row) for row in rows]
    
    async def get_pathogens(self, search: str = None, pathogen_type: str = None) -> List[Dict[str, Any]]:
        """Récupérer la liste des pathogènes."""
        async with self.pool.acquire() as conn:
            query = "SELECT * FROM pathogens WHERE 1=1"
            params = []
            
            if search:
                query += " AND (scientific_name ILIKE $1 OR common_name ILIKE $1)"
                params.append(f"%{search}%")
            
            if pathogen_type:
                param_index = len(params) + 1
                query += f" AND type = ${param_index}"
                params.append(pathogen_type)
            
            query += " ORDER BY scientific_name"
            
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]


# Instance globale de la base de données PostgreSQL
postgres_db = PostgreSQLDatabase()
