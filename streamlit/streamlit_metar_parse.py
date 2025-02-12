import avwx

def parse_metar(metar_string):
    """Convert a METAR string into a user-friendly format."""
    # Extract station ICAO code (first part of METAR string)
    station = metar_string.split()[0]

    # Create Metar object with only the station identifier
    report = avwx.Metar(station)

    # Parse the METAR string manually
    report.parse(metar_string)

    # Extract attributes correctly
    data = report.data  # MetarData object

    # Convert units
    def knots_to_kmh(knots):
        return round(knots * 1.852, 1) if knots else "N/A"

    def knots_to_mph(knots):
        return round(knots * 1.151, 1) if knots else "N/A"

    def celsius_to_fahrenheit(celsius):
        return round((celsius * 9/5) + 32, 1) if celsius is not None else "N/A"

    return {
        "📍 Location": f"{station}",  # Customize if needed
        "🕒 Time": f"{data.time.repr[2:][:2]}:{data.time.repr[2:][2:4]} (UTC)" if data.time else "N/A",
        "🌬️ Wind": (
            f"From {data.wind_direction.repr}° at {data.wind_speed.value} knots "
            f"({knots_to_kmh(data.wind_speed.value)} km/h, {knots_to_mph(data.wind_speed.value)} mph)"
            if data.wind_speed else "Calm"
        ),
        "👀 Visibility": "10+ km (unlimited visibility)" if data.visibility.repr >= "9999" else f"{data.visibility.repr} km",
        "🌡️ Temperature": f"{data.temperature.value}°C ({celsius_to_fahrenheit(data.temperature.value)}°F)" if data.temperature else "N/A",
        "💧 Dew Point": f"{data.dewpoint.value}°C ({celsius_to_fahrenheit(data.dewpoint.value)}°F)" if data.dewpoint else "N/A",
        "📉 Pressure": f"{data.altimeter.value} hPa (Standard pressure: 1013 hPa)" if data.altimeter else "N/A",
        "☁️ Clouds": ", ".join([f"{c.repr[:3]} clouds at {int(c.repr[3:])*100} feet" for c in data.clouds]) if data.clouds else "Clear skies",
    }