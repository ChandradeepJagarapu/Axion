"""
AXION Web Search
----------------
Provides simple internet search capability for AXION.

Uses the DDGS metasearch library.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


class WebSearch:
    """
    Web search engine for AXION.

    Returns a small number of useful search results so that
    the local LLM can use them as context.
    """

    def __init__(
        self,
        max_results: int = 5,
        timeout: int = 10,
    ):
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str) -> list[dict[str, str]]:
        """
        Search the web and return simplified results.
        """

        query = query.strip()

        if not query:
            return []

        try:
            from ddgs import DDGS

            with DDGS(timeout=self.timeout) as ddgs:
                results = ddgs.text(
                    query,
                    region="in-en",
                    safesearch="moderate",
                    max_results=self.max_results,
                )

            cleaned_results: list[dict[str, str]] = []

            for result in results:
                title = str(
                    result.get("title", "")
                ).strip()

                url = str(
                    result.get("href", "")
                ).strip()

                body = str(
                    result.get("body", "")
                ).strip()

                if not title and not body:
                    continue

                cleaned_results.append(
                    {
                        "title": title,
                        "url": url,
                        "body": body,
                    }
                )

            return cleaned_results

        except Exception as exc:
            print(
                f"\nAXION web search error: {exc}",
                flush=True,
            )

            return []
    def get_weather(
        self,
        location: str = "Hyderabad",
    ) -> str:
        """
        Get real current weather using Open-Meteo.
        """

        try:
            import requests

            # Find the location coordinates
            geo_response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": location,
                    "count": 1,
                    "language": "en",
                    "format": "json",
                },
                timeout=self.timeout,
            )

            geo_response.raise_for_status()
            geo_data = geo_response.json()

            locations = geo_data.get("results", [])

            if not locations:
                return (
                    f"I couldn't find weather information "
                    f"for {location}."
                )

            place = locations[0]

            latitude = place["latitude"]
            longitude = place["longitude"]

            place_name = place.get(
                "name",
                location,
            )

            country = place.get(
                "country",
                "",
            )

            # Get actual weather
            weather_response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": (
                        "temperature_2m,"
                        "relative_humidity_2m,"
                        "apparent_temperature,"
                        "precipitation,"
                        "weather_code,"
                        "wind_speed_10m"
                    ),
                    "daily": (
                        "temperature_2m_max,"
                        "temperature_2m_min,"
                        "precipitation_probability_max"
                    ),
                    "timezone": "auto",
                    "forecast_days": 1,
                },
                timeout=self.timeout,
            )

            weather_response.raise_for_status()
            weather = weather_response.json()

            current = weather.get(
                "current",
                {},
            )

            daily = weather.get(
                "daily",
                {},
            )

            temperature = current.get(
                "temperature_2m"
            )

            feels_like = current.get(
                "apparent_temperature"
            )

            humidity = current.get(
                "relative_humidity_2m"
            )

            precipitation = current.get(
                "precipitation"
            )

            wind_speed = current.get(
                "wind_speed_10m"
            )

            weather_code = current.get(
                "weather_code"
            )

            maximum = daily.get(
                "temperature_2m_max",
                [None],
            )[0]

            minimum = daily.get(
                "temperature_2m_min",
                [None],
            )[0]

            rain_probability = daily.get(
                "precipitation_probability_max",
                [None],
            )[0]

            weather_descriptions = {
                0: "clear sky",
                1: "mainly clear",
                2: "partly cloudy",
                3: "overcast",
                45: "foggy",
                48: "foggy",
                51: "light drizzle",
                53: "moderate drizzle",
                55: "dense drizzle",
                61: "light rain",
                63: "moderate rain",
                65: "heavy rain",
                71: "light snow",
                73: "moderate snow",
                75: "heavy snow",
                80: "light rain showers",
                81: "moderate rain showers",
                82: "heavy rain showers",
                95: "thunderstorm",
                96: "thunderstorm with hail",
                99: "thunderstorm with heavy hail",
            }

            description = weather_descriptions.get(
                weather_code,
                "unknown conditions",
            )

            result = (
                f"Weather in {place_name}"
            )

            if country:
                result += f", {country}"

            result += ":\n"

            if temperature is not None:
                result += (
                    f"Current temperature: "
                    f"{temperature}°C\n"
                )

            if feels_like is not None:
                result += (
                    f"Feels like: "
                    f"{feels_like}°C\n"
                )

            result += (
                f"Conditions: {description}\n"
            )

            if humidity is not None:
                result += (
                    f"Humidity: {humidity}%\n"
                )

            if wind_speed is not None:
                result += (
                    f"Wind speed: "
                    f"{wind_speed} km/h\n"
                )

            if precipitation is not None:
                result += (
                    f"Current precipitation: "
                    f"{precipitation} mm\n"
                )

            if maximum is not None:
                result += (
                    f"Today's high: "
                    f"{maximum}°C\n"
                )

            if minimum is not None:
                result += (
                    f"Today's low: "
                    f"{minimum}°C\n"
                )

            if rain_probability is not None:
                result += (
                    f"Rain probability today: "
                    f"{rain_probability}%"
                )

            return result

        except Exception as exc:
            print(
                f"\nAXION weather error: {exc}",
                flush=True,
            )

            return (
                "I couldn't retrieve the current "
                "weather right now."
            )

    def format_results(
        self,
        query: str,
        results: list[dict[str, str]],
    ) -> str:
        """
        Convert search results into compact LLM context.

        Known authoritative domains are placed first so that
        current-information answers are less likely to rely
        on outdated third-party information.
        """

        if not results:
            return ""

        authoritative_domains = {
            "python.org",
            "docs.python.org",
            "devguide.python.org",
            "nodejs.org",
            "github.com",
            "docs.github.com",
            "ubuntu.com",
            "developer.mozilla.org",
            "openai.com",
            "platform.openai.com",
            "developers.google.com",
            "cloud.google.com",
            "mozilla.org",
        }

        def is_authoritative(result: dict[str, str]) -> bool:
            url = result.get("url", "").strip()

            if not url:
                return False

            try:
                hostname = urlparse(url).hostname

                if not hostname:
                    return False

                hostname = hostname.lower()

                if hostname.startswith("www."):
                    hostname = hostname[4:]

                return any(
                    hostname == domain
                    or hostname.endswith("." + domain)
                    for domain in authoritative_domains
                )

            except Exception:
                return False

        prioritized_results = sorted(
            results,
            key=lambda result: (
                not is_authoritative(result)
            ),
        )

        lines = [
            "WEB SEARCH RESULTS",
            f"Search query: {query}",
            "",
            "SOURCE PRIORITY:",
            "Results from known authoritative domains "
            "should generally be preferred over unrelated "
            "third-party sources, especially for current "
            "versions, releases, documentation, and official "
            "product information.",
            "",
        ]

        for index, result in enumerate(
            prioritized_results,
            start=1,
        ):
            title = result.get(
                "title",
                "",
            )

            url = result.get(
                "url",
                "",
            )

            body = result.get(
                "body",
                "",
            )

            priority_label = (
                "AUTHORITATIVE SOURCE"
                if is_authoritative(result)
                else "OTHER SOURCE"
            )

            lines.append(
                f"[{index}] {title}"
            )

            lines.append(
                f"Source type: {priority_label}"
            )

            if body:
                lines.append(
                    f"Summary: {body}"
                )

            if url:
                lines.append(
                    f"Source: {url}"
                )

            lines.append("")

        lines.append(
            "Use the web results as the primary source "
            "for current-information questions. Prefer "
            "authoritative sources when available. If "
            "sources conflict, do not guess; explain the "
            "conflict. Do not invent facts."
        )

        return "\n".join(lines)


def search_web(
    query: str,
    max_results: int = 5,
) -> str:
    """
    Convenience function for AXION.
    """

    searcher = WebSearch(
        max_results=max_results
    )

    results = searcher.search(query)

    return searcher.format_results(
        query,
        results,
    )