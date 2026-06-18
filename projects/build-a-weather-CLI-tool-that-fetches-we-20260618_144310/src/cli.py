from typing import Dict
import argparse
from src.config import Config
from src.weather import WeatherClient

def cli() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Weather CLI tool")
    parser.add_argument("city", help="City name")
    parser.add_argument("-f", "--forecast", action="store_true", help="Get forecast")
    args = parser.parse_args()

    config = Config(**Config.load_config())
    client = WeatherClient(config)

    if args.forecast:
        forecast = client.get_forecast(args.city)
        print(f"Weather forecast for {args.city}:")
        for forecast_data in forecast["list"]:
            print(f"Date: {forecast_data['dt_txt']}, Temp: {forecast_data['main']['temp']}")
    else:
        weather = client.get_weather(args.city)
        print(f"Weather in {args.city}:")
        print(f"Temp: {weather['main']['temp']}, Humidity: {weather['main']['humidity']}")

if __name__ == "__main__":
    cli()