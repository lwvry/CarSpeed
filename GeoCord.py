import requests
import pyproj
import math
import numpy as np

def transform_coordinates(coordinates, source_epsg="EPSG:4326", target_epsg="EPSG:2180"):
    """
    Przekształcanie współrzędnych z jednego układu na inny
    """
    transformer = pyproj.Transformer.from_crs(source_epsg, target_epsg, always_xy=True, accuracy=1.0)
    return [transformer.transform(longtitude, latitude) for latitude, longtitude in coordinates]

def calculate_bbox(coordinates):
    """
    Obliczanie powierchni dla podanych współrzędnych
    """
    x_coords, y_coords = zip(*coordinates)
    if (max(x_coords) - min(x_coords) < 1 or max(y_coords) - min(y_coords) < 1):
        return(min(x_coords), min(y_coords), max(x_coords)+1, max(y_coords)+1) # Numeryczny model terenu nie pobierze się dla wymiarów 1x1
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

def calculate_dimensions(bbox):
    """
    Obliczanie szerokości i wysokości tak, aby zapewnić dokładność 1 metra
    """
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    width = math.ceil(bbox_width)
    height = math.ceil(bbox_height)
    return width, height

def generate_wcs_url(bbox, width, height, coverage_id="DTM_PL-EVRF2007-NH", crs="EPSG:2180"):
    """
    Generowanie URL w celu pobrania danych wysokości
    """
    base_url = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMT/GRID1/WCS/DigitalTerrainModel"
    params = {
        "SERVICE": "WCS",
        "VERSION": "1.0.0",
        "REQUEST": "GetCoverage",
        "FORMAT": "image/x-aaigrid",
        "COVERAGE": coverage_id,
        "BBOX": ",".join(map(str, bbox)),
        "CRS": crs,
        "RESPONSE_CRS": crs,
        "WIDTH": width,
        "HEIGHT": height
    }
    response_url = f"{base_url}?" + "&".join([f"{key}={value}" for key, value in params.items()])
    return response_url

def download_ascii_grid(url, output_file="heights.asc"):
    """
    Pobranie numerycznego modelu terenu z podanego URL i zapis do pliku
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_file, "w") as file:
            file.write(response.text)
        print(f"Downloaded NMT")
    else:
        raise Exception(f"Download error: {response.status_code}, {response.text}")

def parse_ascii_grid(file_path, coordinates, bbox):
    """
    Przypisanie wysokości odpowiednio dla listy lokalizacji drona
    """
    try:
        with open(file_path, 'r') as file:
            header = {}
            for _ in range(6):
                line = file.readline().strip()
                key, value = line.split()
                header[key.lower()] = float(value)
            data = np.loadtxt(file) # Wczytanie danych wysokości w formie macierzy
        heights = []
        for x, y in coordinates:
            col = int(math.floor(x - bbox[0]))
            row = int(math.floor(bbox[3] - y))
            heights.append(data[row, col])  # Pobranie wysokości dla danej współrzędnej
        return heights
    except Exception as e:
        print(f"Error parsing ASCII Grid file: {e}")
        return []