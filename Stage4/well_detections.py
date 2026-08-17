import requests #HTTP Requests
from math import radians, sin, cos, sqrt, atan2, degrees

def haversine_distance(lat1, lon1, lat2, lon2): #https://www.movable-type.co.uk/scripts/latlong.html?from=48.7819900%2C-122.2936380&to=48.7761100%2C-122.3395200
    R = 6371.0

    lat1 = radians(lat1)
    lat2 = radians(lat2)

    dlat = lat2 - lat1
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2)**2
        + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    )

    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def bearing_from_well_to_plume(
    well_lat, well_lon,
    plume_lat, plume_lon
):
    lat1 = radians(well_lat)
    lat2 = radians(plume_lat)
    dlon = radians(plume_lon - well_lon)

    x = sin(dlon) * cos(lat2)

    y = (
        cos(lat1) * sin(lat2)
        - sin(lat1) * cos(lat2) * cos(dlon)
    )

    bearing = degrees(atan2(x, y))

    return (bearing + 360) % 360

plume_lat = 32 
plume_lon = -103.9
buffer = 0.02

lat_min = plume_lat - buffer
lat_max = plume_lat + buffer
lon_min = plume_lon - buffer
lon_max = plume_lon + buffer

url = "https://data.texas.gov/resource/uumf-5r4y.json"

params = {
    "$select": "api,gis_lat83,gis_long83",
    "$where": (
        f"gis_lat83 >= {lat_min} "
        f"AND gis_lat83 <= {lat_max} "
        f"AND gis_long83 >= {lon_min} "
        f"AND gis_long83 <= {lon_max}"
    ),
    "$limit": 50000
}

response = requests.get(url, params=params)

print(response.url)
print(response.status_code)

data = response.json()

print(f"Found {len(data)} wells")

candidates = [
    {
        "api": well["api"],
        "lat": float(well["gis_lat83"]),
        "lon": float(well["gis_long83"])
    }
    for well in data
]
for candidate in candidates:
    candidate["distance_km"] = haversine_distance(
        candidate["lat"],
        candidate["lon"],
        plume_lat,
        plume_lon
    )
for candidate in candidates:
    candidate["bearing_to_plume"] = bearing_from_well_to_plume(
        candidate["lat"],
        candidate["lon"],
        plume_lat,
        plume_lon
    )