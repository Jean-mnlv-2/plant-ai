"""
Service météo pour l'API Plant-AI.
"""
import requests
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from .models import (
    WeatherResponse, WeatherLocation, Coordinates, CurrentWeather,
    WeatherForecast, WeatherAlert, AgriculturalConditions,
    WeatherCondition, SeverityLevel, Temperature, Precipitation
)


class WeatherService:
    """Service pour récupérer les données météo agricoles."""
    
    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY", "demo_key")
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    def get_weather_data(self, lat: float, lng: float, include_forecast: bool = True) -> WeatherResponse:
        """Récupérer les données météo pour une localisation."""
        try:
            # Données météo actuelles
            current_weather = self._get_current_weather(lat, lng)
            
            # Prévisions météo
            forecast = []
            if include_forecast:
                forecast = self._get_weather_forecast(lat, lng)
            
            # Alertes météo
            alerts = self._get_weather_alerts(lat, lng)
            
            # Conditions agricoles
            agricultural_conditions = self._get_agricultural_conditions(current_weather, forecast)
            
            return WeatherResponse(
                success=True,
                location=WeatherLocation(
                    name=self._get_location_name(lat, lng),
                    country=self._get_country_name(lat, lng),
                    coordinates=Coordinates(latitude=lat, longitude=lng)
                ),
                current=current_weather,
                forecast=forecast,
                alerts=alerts,
                agriculturalConditions=agricultural_conditions
            )
            
        except Exception as e:
            # Retourner des données de démonstration en cas d'erreur
            return self._get_demo_weather_data(lat, lng)
    
    def _get_current_weather(self, lat: float, lng: float) -> CurrentWeather:
        """Récupérer les conditions météo actuelles."""
        if self.api_key == "demo_key":
            return self._get_demo_current_weather()
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                "lat": lat,
                "lon": lng,
                "appid": self.api_key,
                "units": "metric",
                "lang": "fr"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return CurrentWeather(
                temperature=data["main"]["temp"],
                humidity=data["main"]["humidity"],
                windSpeed=data["wind"]["speed"],
                windDirection=self._get_wind_direction(data["wind"].get("deg", 0)),
                visibility=data.get("visibility", 10000) / 1000,  # Convert to km
                pressure=data["main"]["pressure"],
                uvIndex=self._get_uv_index(lat, lng),
                condition=self._map_weather_condition(data["weather"][0]["main"]),
                description=data["weather"][0]["description"],
                timestamp=datetime.utcnow()
            )
        except Exception:
            return self._get_demo_current_weather()
    
    def _get_weather_forecast(self, lat: float, lng: float) -> List[WeatherForecast]:
        """Récupérer les prévisions météo."""
        if self.api_key == "demo_key":
            return self._get_demo_forecast()
        
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "lat": lat,
                "lon": lng,
                "appid": self.api_key,
                "units": "metric",
                "lang": "fr"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Grouper par jour et prendre les prévisions pour les 5 prochains jours
            daily_forecasts = {}
            for item in data["list"][:40]:  # 5 jours * 8 prévisions par jour
                date = item["dt_txt"].split(" ")[0]
                if date not in daily_forecasts:
                    daily_forecasts[date] = []
                daily_forecasts[date].append(item)
            
            forecasts = []
            for date, items in list(daily_forecasts.items())[:5]:
                temps = [item["main"]["temp"] for item in items]
                humidity = sum(item["main"]["humidity"] for item in items) // len(items)
                wind_speed = sum(item["wind"]["speed"] for item in items) // len(items)
                condition = self._map_weather_condition(items[0]["weather"][0]["main"])
                
                forecasts.append(WeatherForecast(
                    date=date,
                    temperature=Temperature(min=min(temps), max=max(temps)),
                    humidity=humidity,
                    windSpeed=wind_speed,
                    condition=condition,
                    precipitation=Precipitation(
                        probability=self._get_precipitation_probability(items),
                        amount=self._get_precipitation_amount(items)
                    ),
                    agriculturalAdvice=self._get_agricultural_advice(condition, min(temps), max(temps))
                ))
            
            return forecasts
        except Exception:
            return self._get_demo_forecast()
    
    def _get_weather_alerts(self, lat: float, lng: float) -> List[WeatherAlert]:
        """Récupérer les alertes météo."""
        # Pour la démo, générer quelques alertes basées sur les conditions
        alerts = []
        
        # Simuler une alerte de gelée si la température est basse
        current_temp = self._get_demo_current_weather().temperature
        if current_temp < 5:
            alerts.append(WeatherAlert(
                type="frost_warning",
                severity=SeverityLevel.MEDIUM,
                title="Risque de gelée nocturne",
                description="Températures prévues entre 0°C et 2°C dans la nuit",
                validFrom=datetime.utcnow() + timedelta(hours=2),
                validTo=datetime.utcnow() + timedelta(hours=8),
                recommendations=[
                    "Protéger les cultures sensibles",
                    "Arroser légèrement avant le coucher du soleil"
                ]
            ))
        
        return alerts
    
    def _get_agricultural_conditions(self, current: CurrentWeather, forecast: List[WeatherForecast]) -> AgriculturalConditions:
        """Déterminer les conditions agricoles optimales."""
        # Logique basique pour déterminer les conditions
        temp_ok = 15 <= current.temperature <= 30
        humidity_ok = 40 <= current.humidity <= 80
        wind_ok = current.windSpeed < 20
        
        irrigation = "not_needed" if current.humidity > 60 else "needed"
        spraying = "favorable" if temp_ok and wind_ok else "not_favorable"
        harvesting = "favorable" if temp_ok and not current.condition == WeatherCondition.RAINY else "not_favorable"
        planting = "favorable" if temp_ok and humidity_ok else "not_favorable"
        
        return AgriculturalConditions(
            irrigation=irrigation,
            spraying=spraying,
            harvesting=harvesting,
            planting=planting
        )
    
    def _get_location_name(self, lat: float, lng: float) -> str:
        """Obtenir le nom de la localisation."""
        # Pour la démo, retourner des noms basés sur les coordonnées
        if 48.0 <= lat <= 49.0 and 2.0 <= lng <= 3.0:
            return "Paris"
        elif 45.0 <= lat <= 46.0 and 4.0 <= lng <= 5.0:
            return "Lyon"
        else:
            return f"Location {lat:.2f}, {lng:.2f}"
    
    def _get_country_name(self, lat: float, lng: float) -> str:
        """Obtenir le nom du pays."""
        # Logique basique pour la démo
        if 42.0 <= lat <= 51.0 and -5.0 <= lng <= 8.0:
            return "France"
        else:
            return "Unknown"
    
    def _get_wind_direction(self, degrees: float) -> str:
        """Convertir les degrés en direction du vent."""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
        index = int((degrees + 11.25) / 22.5) % 16
        return directions[index]
    
    def _get_uv_index(self, lat: float, lng: float) -> int:
        """Obtenir l'index UV (simulation)."""
        # Simulation basique basée sur la latitude et l'heure
        import math
        hour = datetime.utcnow().hour
        uv_base = max(0, 10 - abs(lat - 45) * 0.2)
        uv_hour = max(0, uv_base * math.sin((hour - 6) * math.pi / 12))
        return min(11, int(uv_hour))
    
    def _map_weather_condition(self, condition: str) -> WeatherCondition:
        """Mapper les conditions météo OpenWeatherMap vers nos enums."""
        mapping = {
            "Clear": WeatherCondition.SUNNY,
            "Clouds": WeatherCondition.CLOUDY,
            "Rain": WeatherCondition.RAINY,
            "Thunderstorm": WeatherCondition.STORMY,
            "Snow": WeatherCondition.SNOWY,
            "Mist": WeatherCondition.FOGGY,
            "Fog": WeatherCondition.FOGGY
        }
        return mapping.get(condition, WeatherCondition.CLOUDY)
    
    def _get_precipitation_probability(self, items: List[Dict]) -> int:
        """Calculer la probabilité de précipitation."""
        if not items:
            return 0
        return sum(item.get("pop", 0) * 100 for item in items) // len(items)
    
    def _get_precipitation_amount(self, items: List[Dict]) -> float:
        """Calculer la quantité de précipitation."""
        if not items:
            return 0.0
        total = sum(item.get("rain", {}).get("3h", 0) for item in items)
        return total / len(items)
    
    def _get_agricultural_advice(self, condition: WeatherCondition, min_temp: float, max_temp: float) -> str:
        """Générer des conseils agricoles basés sur les conditions."""
        if condition == WeatherCondition.RAINY:
            return "Éviter les traitements foliaires, conditions humides"
        elif condition == WeatherCondition.SUNNY and max_temp > 25:
            return "Conditions idéales pour l'irrigation et les traitements"
        elif min_temp < 5:
            return "Protéger les cultures sensibles au froid"
        else:
            return "Conditions favorables pour l'agriculture"
    
    def _get_demo_weather_data(self, lat: float, lng: float) -> WeatherResponse:
        """Données météo de démonstration."""
        return WeatherResponse(
            success=True,
            location=WeatherLocation(
                name=self._get_location_name(lat, lng),
                country=self._get_country_name(lat, lng),
                coordinates=Coordinates(latitude=lat, longitude=lng)
            ),
            current=self._get_demo_current_weather(),
            forecast=self._get_demo_forecast(),
            alerts=self._get_weather_alerts(lat, lng),
            agriculturalConditions=AgriculturalConditions(
                irrigation="not_needed",
                spraying="favorable",
                harvesting="favorable",
                planting="favorable"
            )
        )
    
    def _get_demo_current_weather(self) -> CurrentWeather:
        """Conditions météo actuelles de démonstration."""
        return CurrentWeather(
            temperature=25.0,
            humidity=70,
            windSpeed=12.0,
            windDirection="NW",
            visibility=10.0,
            pressure=1013.0,
            uvIndex=6,
            condition=WeatherCondition.SUNNY,
            description="Ensoleillé - Idéal pour l'agriculture",
            timestamp=datetime.utcnow()
        )
    
    def _get_demo_forecast(self) -> List[WeatherForecast]:
        """Prévisions météo de démonstration."""
        forecasts = []
        for i in range(5):
            date = (datetime.utcnow() + timedelta(days=i+1)).strftime("%Y-%m-%d")
            forecasts.append(WeatherForecast(
                date=date,
                temperature=Temperature(min=18.0, max=26.0),
                humidity=65,
                windSpeed=8.0,
                condition=WeatherCondition.CLOUDY if i % 2 == 0 else WeatherCondition.SUNNY,
                precipitation=Precipitation(probability=20, amount=0.0),
                agriculturalAdvice="Conditions favorables pour les traitements foliaires"
            ))
        return forecasts


# Instance globale du service météo
weather_service = WeatherService()


