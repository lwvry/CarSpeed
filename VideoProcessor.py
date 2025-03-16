import cv2
import math
from ultralytics import YOLO
from CarContainer import CarContainer
import torch
import re
import os
import numpy as np
import csv
from GeoCord import (
    transform_coordinates,
    calculate_bbox,
    calculate_dimensions,
    generate_wcs_url,
    download_ascii_grid,
    parse_ascii_grid
)
# Słownik z parametrami dronów
DRONES = {
    "DJI mini 4 pro": {
        "focal_length": 6.7,
        "sensor_width": 8.9739,
        "sensor_height": 6.7175
    },
    "DJI air 2s": {
        "focal_length": 22,
        "sensor_width": 13.2,
        "sensor_height": 8.8
    },
}

class VideoProcessor:
    """
    Klasa odpowiedzialna za przetwarzanie klatek nagrania i detekcję pojazdów
    """
    def __init__(self, video_path, drone_model, altitude, model_path):
        """
        Args:
            video_path: Ścieżka do pliku wideo
            drone_model: Model drona
            altitude: Początkowa wysokość drona
            model_path: Ścieżka do modelu YOLO
        """
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Failed to open the video")
        self.start_altitude = altitude

        # Pobranie parametrów dla zadanego modelu drona
        self.drone = self._select_drone(drone_model)
        self.focal_length = self.drone["focal_length"]
        self.sensor_width = self.drone["sensor_width"]
        self.sensor_height = self.drone["sensor_height"]

        self.height_file = "heights.asc"    # Plik z numerycznym modelem terenu
        self.output_file = "traffic_analysis.csv"   # Plik do zapisu danych z analizy

        with open(self.output_file, 'w') as file:   # Usuwanie zawartości i zapis nagłówka
            writer = csv.writer(file)
            writer.writerow(["Time (s)", "Avg Speed (km/h)", "Traffic Density"])

        # Wybór urządzenia
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
            )
        # Załadowanie modelu YOLO
        try:
            self.model = YOLO(model_path,verbose=False).to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to load the model: {e}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,5)

        srt_path = self._get_srt_path(video_path)
        self.latitude = self._parse_srt_field(srt_path, r"\[latitude:\s*([\d.]+)\]")    # Odczytanie szerokości geograficznej
        self.longitude = self._parse_srt_field(srt_path, r"\[longitude:\s*([\d.]+)\]")  # Odczytanie długości geograficznej

        
        try:    # Sprawdzenie czy dane są zapisane jako altitude, a jezeli nie to odczytywane są dane z rel_alt
            self.altitudes = self._parse_srt_field(srt_path, r"\[altitude:\s*([\d.]+)\]")
        except ValueError:
            self.altitudes = self._parse_srt_field(srt_path, r"\[rel_alt:\s*([\d.]+)")
        if len(self.latitude) < self.total_frames:
            self.latitude += [self.latitude[-1]] * (self.total_frames - len(self.latitude))
        if len(self.longitude) < self.total_frames:
            self.longitude += [self.longitude[-1]] * (self.total_frames - len(self.longitude))
        if len(self.altitudes) < self.total_frames:
            self.altitudes += [self.altitudes[-1]] * (self.total_frames - len(self.altitudes))

        # Obliczanie rzeczywistych wysokości drona
        self.coordinates = list(zip(self.latitude, self.longitude))
        self.real_altitudes = np.array(self._fetch_real_altitudes(self.height_file))
        if self.start_altitude:
            self.real_altitudes[0] = self.start_altitude
        if self.real_altitudes[0]:
            self.real_altitudes = (np.array(self.altitudes) + (self.real_altitudes[0] - self.real_altitudes)).tolist()  # Korekta wysokości
        else:
            self.real_altitudes = self.altitudes

        # Inicjalizacja kontenera do śledzenia pojazdów
        self.car_container = CarContainer(
            self.fps, self.frame_width, self.frame_height,
            self.focal_length, self.sensor_width, self.sensor_height,max_frames_missing=10
        )
        self.current_frame_idx = 0
    
    def _select_drone(self, model_name):
        if model_name in DRONES:
            return DRONES[model_name]
        raise ValueError(f"Model {model_name} not found")

    def _parse_srt_field(self, srt_path, pattern):
        """
        Odczyt danych z pliku SRT na podstawie wzorca
        """
        values = []
        try:
            with open(srt_path, 'r') as file:
                for line in file:
                    match = re.search(pattern, line)
                    if match:
                        value = float(match.group(1))
                        values.append(value)
        except FileNotFoundError:
            raise ValueError(f"SRT file not found: {srt_path}")

        if len(values) == 0:
            raise ValueError(f"No data found in the SRT file")
        return values

    def _get_srt_path(self, video_path):
        """
        Odczyt ścieżki do pliku SRT na podstawie ścieżki do nagrania (nazwa nagrania + rozszerzenie .srt)
        """
        srt_path = os.path.splitext(video_path)[0] + ".srt"
        if not os.path.isfile(srt_path):
            raise FileNotFoundError(f"No .srt file found for video: {video_path}")
        return srt_path
    
    def _fetch_real_altitudes(self, output_file):
        """
        Pobieranie rzeczywistej wysokości na podstawie współrzędnych
        """
        try:
            transformed_coords = transform_coordinates(self.coordinates, source_epsg="EPSG:4326", target_epsg="EPSG:2180")
            bbox = calculate_bbox(transformed_coords)
            width, height = calculate_dimensions(bbox)
            if not self.start_altitude:
                if(width < 5 and height < 5):   # Ograniczenie do +- 5 metrów poziomo lub pionowo, aby niepotrzebnie pobierać numeryczny model terenu
                    heights = [0] * len(transformed_coords)
                    return heights
            # Jezeli przekroczono zakres lub podano wysokość startową następuje pobranie numerycznego modelu terenu
            url = generate_wcs_url(bbox, width, height)
            download_ascii_grid(url, output_file)
            heights = parse_ascii_grid(output_file, transformed_coords, bbox)
    
            return heights
        except Exception as e:
            print(f"Error fetching real altitudes: {e}")
            return []

    def process_frame(self):
        """
        Przetwarzanie pojedynczczej klatki w celu detekcji obiektów
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, False
        
        drone_real_height = self.real_altitudes[self.current_frame_idx]
        self.car_container.update_drone_height(drone_real_height)
        self.car_container.increment_missing_frames()   # Inkrementacja licznika zgubionych pozycji dla kazdego pojazdu

        self.current_frame_idx += 1
        results_t = self.model(frame, conf=0.70, imgsz=1280, stream=False, verbose = False)
        results = [t.cpu().numpy() for t in results_t]
        # Przetwarzanie rezultatów detekcji
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None:
                for box in result.obb:
                    x_center, y_center, width, height, theta = box.xywhr[0]
                    class_id = int(box.cls[0])
                    vehicle_type = {9: 'large', 10: 'small'}.get(class_id)  # Sprawdzenie czy wykryty obiekt nalezy do klasy 9 albo 10
                    if not vehicle_type:
                        continue

                    position = (x_center, y_center, width, height, math.degrees(theta))
                    self.car_container.update_or_add_car(position, vehicle_type)     # Aktualizacja pozycji lub dodanie nowego pojazdu

        if len(self.car_container.cars) > 100:
            self.car_container.cars = self.car_container.cars[-100:]
        self.car_container.remove_missing_cars()    # Usunięcie zgubionych pojazdów
        if self.current_frame_idx % round(self.fps) == 0:
            self.avg_speed_and_traffic(self.output_file)    # Zapis do pliku informacji co sekundę nagrania
        processed_frame = self.car_container.draw_cars(frame)
        return processed_frame, True

    def avg_speed_and_traffic(self, output_filepath):
        seconds = self.current_frame_idx / self.fps
        detected_cars = [car for car in self.car_container.cars if car.is_detected and len(car.real_speed_history) > 2]
        if not detected_cars:
            avg_speed = 0.0
        else:
            avg_speed = np.mean([car.real_speed for car in detected_cars])
        traffic = len(detected_cars)
        with open(output_filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([round(seconds), round(avg_speed, 2), traffic])


    def get_total_frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_speed_history(self, car_id):
        return self.car_container.get_speed_history(car_id)