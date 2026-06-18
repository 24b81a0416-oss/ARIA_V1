from typing import Dict
import requests
from src.config import Config

class WeatherClient:
    """Weather API client."""
    def __init__(self, config: Config):
        self.config = config

    def get_weather(self, city: str) -> Dict:
        """Get the weather for a given city."""
        params = {
            "q": city,
            "appid": self.config.get_api_key(),
            "units": "metric"
        }
        response = requests.get(f"{self.config.get_base_url()}/weather", params=params)
        return response.json()

    def get_forecast(self, city: str) -> Dict:
        """Get the forecast for a given city."""
        params = {
            "q": city,
            "appid": self.config.get_api_key(),
            "units": "metric"
        }
        response = requests.get(f"{self.config.get_base_url()}/forecast", params=params)
        return response.json()