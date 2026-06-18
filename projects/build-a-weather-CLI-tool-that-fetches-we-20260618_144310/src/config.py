from typing import Dict

class Config:
    """Project configuration."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"

    def get_api_key(self) -> str:
        """Get the API key."""
        return self.api_key

    def get_base_url(self) -> str:
        """Get the base URL."""
        return self.base_url

    @staticmethod
    def load_config() -> Dict:
        """Load configuration from environment variables."""
        import os
        return {
            "api_key": os.environ.get("OPENWEATHER_API_KEY"),
        }