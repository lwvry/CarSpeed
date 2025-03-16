import cv2
import numpy as np
from Car import Car

class CarContainer:
    """
    Kontener do śledzenia i zarządzania wykrytymi pojazdami
    """
    def __init__(self, fps, frame_width, frame_height, focal_length, sensor_width, sensor_height, max_frames_missing=3):
        """
        Args:
            fps: Liczba klatek na sekundę
            frame_width: Szerokość klatki w pikselach
            frame_height: Wysokość klatki w pikselach
            focal_length: Ogniskowa kamery
            sensor_width: Szerokość sensora kamery w milimetrach
            sensor_height: Wysokość sensora kamery w milimetrach
            max_frames_missing: Maksymalna liczba klatek, w których pojazd może być zgubiony
        """
        self.cars = []  # Lista śledzonych pojazdów
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focal_length = focal_length

        # Przycinanie wymiarów sensora, aby dopasować je do proporcji 16:9
        self.sensor_width, self.sensor_height = self._crop_dimensions(sensor_width, sensor_height, 16/9)
        self.max_frames_missing = max_frames_missing
        self.next_id = 1
        self.car_counter = 0
        self.region = self._get_centered_region()   

    def _get_centered_region(self):
        """
        Obliczanie regionu nagrania, w którym będą śledzone pojazdy
        """
        region_width = int(self.frame_width * 0.95)
        region_height = int(self.frame_height * 0.95)
        x1 = (self.frame_width - region_width) // 2
        y1 = (self.frame_height - region_height) // 2
        return ((x1, y1), (x1 + region_width, y1 + region_height))

    def _crop_dimensions(self, width, height, aspect_ratio):
        if height * aspect_ratio <= width:
            return int(height * aspect_ratio), height
        return width, int(width / aspect_ratio)

    def update_drone_height(self, drone_real_height):
        """
        Aktualizacja wysokości drona i obliczanie GSD
        """
        self.drone_real_height = drone_real_height
        self.gsd_horizontal = (drone_real_height * self.sensor_width) / (self.focal_length * self.frame_width)
        self.gsd_vertical = (drone_real_height * self.sensor_height) / (self.focal_length * self.frame_height)
        Car.scale = np.array([self.gsd_horizontal, self.gsd_vertical])

    def update_or_add_car(self, new_position, vehicle_type):
        """
        Aktualizacja pozycji istniejącego pojazdu lub dodanie nowego
        """
        (x1, y1), (x2, y2) = self.region
        x_center, y_center = new_position[:2]
        if not (x1 <= x_center <= x2 and y1 <= y_center <= y2): # Sprawdzenie, czy pojazd znajduje się w regionie śledzenia
            return
        
        for car in self.cars:
            # Ustawienie progu odległości w zależności od liczby zgubionych pozycji i ilości pozycji w historii
            distance_threshold = 20 if car.frames_since_seen <=1 and len(car.positions_history) > 5 else 30
            # Sprawdzenie, czy nowa pozycja znajduje się w pobliżu przewidywanej pozycji
            if np.linalg.norm(np.array(new_position[:2]) - np.array(car.predict_next_position()[:2])) < distance_threshold:
                car.update_position(new_position)    # Sprawdzenie, czy nowa pozycja znajduje się w pobliżu przewidywanej pozycji
                car.calculate_speed(self.fps)   # Obliczanie prędkości
                # Sprawdzenie, czy pojazd został wykryty i czy jego prędkość jest większa niż 10 km/h
                if not car.is_detected and car.real_speed > 10:
                    car.is_detected = True
                    self.car_counter += 1   # Zwiększenie licznika pojazdów
                return
            
        # Dodanie nowego pojazdu
        new_car = Car(new_position, vehicle_type)
        new_car.id = self.next_id
        self.next_id += 1
        self.cars.append(new_car)

    def remove_missing_cars(self):
        """
        Usuwanie pojazdów, które zniknęły
        """
        self.cars = [
            car for car in self.cars
            if car.frames_since_seen < self.max_frames_missing  # Warunek usunięcia
        ]

    def increment_missing_frames(self):
        """
        Zwiększa licznik zgubionych pozycji każdego pojazdu.
        """
        for car in self.cars:
            car.increment_frames_since_seen()

    def draw_cars(self, frame):
        """
        Rysowanie pojazdów na klatce nagrania
        """
        scale = 2

        # Rysowanie wysokości drona
        text = f"Altitude: {self.drone_real_height:.1f} m"
        cv2.putText(frame, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 6 * scale, cv2.LINE_AA)
        cv2.putText(frame, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (189, 114, 0), 2 * scale, cv2.LINE_AA)

        # Rysowanie licznika pojazdów
        text = f"Car Counter: {self.car_counter}"
        cv2.putText(frame, text, (600, 80), cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 6 * scale, cv2.LINE_AA)
        cv2.putText(frame, text, (600, 80), cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (189, 114, 0), 2 * scale, cv2.LINE_AA)    
        
        # Rysowanie ramek i ściezki ruchu dla pojazdów
        for car in self.cars:
            if not car.is_detected:
               continue

            # Rysowanie ścieżki ruchu pojazdu
            if len(car.approximated_positions) > 1:
                points = [tuple(map(int, pos[:2])) for pos in car.approximated_positions]
                for p1, p2 in zip(points[:-1], points[1:]):
                    cv2.line(frame, p1, p2, [48, 172, 119], 3 * scale)

            x_center, y_center, width, height, theta = map(float, car.position)
            rect = ((x_center, y_center), (width, height), theta)
            box_points = cv2.boxPoints(rect).astype(int)    # Konwersja z (x, y, szerokość, wysokość, kąt) na pozycje wierzchołków prostokąta 
            color = [189, 114, 0] if car.vehicle_type == 'small' else [25, 83, 217]
            cv2.drawContours(frame, [box_points], 0, color, 3 * scale)

            #Przygotowanie danych o pojazdach
            car_data = f"ID: {car.id} | Speed: {car.real_speed:.0f} km/h"
            additional_info = f"Type: {car.vehicle_type} Lost: {car.frames_since_seen}"
            
            car_data_position = (int(x_center - 300), int(y_center - 60 * scale))
            additional_info_position = (int(x_center - 300), int(y_center - 30 * scale))
            
            # Wyświetlanie danych o pojazdach
            cv2.putText(frame, car_data, car_data_position, cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 6 * scale, cv2.LINE_AA)
            cv2.putText(frame, car_data, car_data_position, cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, color, 2 * scale, cv2.LINE_AA)

            cv2.putText(frame, additional_info, additional_info_position, cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 6 * scale, cv2.LINE_AA)
            cv2.putText(frame, additional_info, additional_info_position, cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, color, 2 * scale, cv2.LINE_AA)

        return frame

    def get_car_by_id(self, car_id):
        return next((car for car in self.cars if car.id == car_id and car.is_detected), None)


    def get_speed_history(self, car_id):
        car = self.get_car_by_id(car_id)
        return car.real_speed_history if car else None

    def get_traffic_density(self):
        detected_cars = [car for car in self.cars if car.is_detected]
        return len(detected_cars) / max(1, self.fps)
