import numpy as np
class Car:
    """
    Klasa reprezentująca pojedyńczy pojazd
    """
    scale = None    # Poziome i pionowe GSD do przeliczania pikseli na metry
    def __init__(self, position, vehicle_type):
        """
        Args:
            position: Pozycja pojazdu oraz jego wymiary (x, y, szerokość, wysokość, kąt)
            vehicle_type: Typ pojazdu ("small" lub "large")
        """
        self.id = None
        self.is_detected = False
        self.position = position    # Aktualna pozycja pojazdu
        self.positions_history = [position] # Historia pozycji pojazdu
        self.approximated_positions = []    # Pozycje wyznaczone na podstawie aproksymacji ruchu
        self.speed = 0.0 # Prędkość w km/h
        self.speed_history = [] # Historia prędkości
        self.real_speed = 0.0   # Prędkość uśredniona z 10 pomiarów
        self.real_speed_history = []    # Historia uśrednionej prędkości
        self.vehicle_type = vehicle_type    # Typ pojazdu
        self.frames_since_seen = 0  # Liczba zgubionych pozycji dla pojazdu
        self.detection_counter = 0  # Licznik klatek do uśredniania prędkości 
    
    def update_position(self, new_position):
        """
        Aktualizacja pozycji pojazdu i reset licznika zgubionych pozycji
        """
        # Dodanie nowej pozycji do historii pozycji
        self.positions_history.append(new_position)
        if len(self.positions_history) > 10:
            self.positions_history = self.positions_history[-10:]   # Zachowanie ostatnich 10 pozycji

        # Aktualizacja aktualnej pozycji
        self.position = new_position
        self._update_approximated_positions()

        # Jeśli pojazd był zgubiony, to następuje poprawa histori pozycji
        if self.frames_since_seen > 1 and len(self.positions_history) >= 10:
            self.positions_history.pop()
            aproximated_pos = self.approximated_positions[:-1]
            aproximated_pos.pop()
            aproximated_pos = [p + new_position[2:] for p in aproximated_pos]
            last_values = aproximated_pos[-self.frames_since_seen+1:]    
            self.positions_history += last_values     
            self.positions_history.append(new_position)
            self.positions_history = self.positions_history[-10:]

        # Reset licznika zgubionych pozycji
        self.frames_since_seen = 0

    def _update_approximated_positions(self):
        """
        Wyznaczenie aproksymowanych pozycji pojazdu
        """
        if len(self.positions_history) < 3: # jezeli za mało danych to zwraca historię pozycji (tylko współrzędne x i y środka pojazdu)
            self.approximated_positions = [pos[:2] for pos in self.positions_history]
            return

        time_indices = np.arange(len(self.positions_history))
        if len(self.positions_history) > 3:
            time_indices[-1] += self.frames_since_seen - 1 # Zwiększenie o zgubione pozycje
        positions_array = np.array(self.positions_history)

        # Dopasowanie wielomianu dla współrzędnych x i y
        coeffs_x = np.polyfit(time_indices, positions_array[:, 0], 2)
        coeffs_y = np.polyfit(time_indices, positions_array[:, 1], 2)
    
        time_indices = np.arange(len(self.positions_history)) + self.frames_since_seen - 1  # Odpowiednie wyrówananie o zgubione pozycje

        # Obliczenie aproksymowanych pozycji
        approx_x = np.polyval(coeffs_x, time_indices)
        approx_y = np.polyval(coeffs_y, time_indices)

        self.approximated_positions = list(zip(approx_x, approx_y))

    def _calculate_average_velocity(self, history_length=10):
        """
        Obliczanie prędkości pojazdu na podstawie pozycji i skali
        """
        if len(self.approximated_positions) < history_length:   # Minimum 10 pozycji aby obliczyć prędkość
            return np.array([0.0, 0.0])

        relevant_positions = np.array(self.approximated_positions[-history_length:])
        velocity_vectors = np.diff(relevant_positions, axis=0)  # Obliczanie róznic pomiędzy pozycjami
        return np.mean(velocity_vectors, axis=0) # Uśrednianie wartości

    def calculate_speed(self, fps):
        if len(self.positions_history) < 10:
            return

        average_velocity = self._calculate_average_velocity(history_length=10)  # Obliczenie średniej prędkości w pikselach
        speed_m_per_s = np.linalg.norm(average_velocity * Car.scale) * fps #Obliczenie długości wektora prędkości
        self.speed = speed_m_per_s * 3.6 # Prędkość w km/h
        
        # Aktualizacja historii prędkości
        self.speed_history.append(self.speed)
        if len(self.speed_history) > 10:
            self.speed_history.pop(0)
        
        self.detection_counter += 1
        if self.detection_counter >= 10:    # Aktualizacja prędkości co 10 detekcji
            self.real_speed = np.mean(self.speed_history)
            self.real_speed_history.append(self.real_speed)
            self.detection_counter = 0

    def predict_next_position(self):
        """
        Przewidywanie następnej pozycji pojazdu na podstawie aproksymacji
        """
        if len(self.positions_history) < 2: # Zwracanie pozycji, gdy za mało danych do aproksymacji
            return self.position
        
        positions_array = np.array(self.positions_history)
        if len(positions_array) == 2:
            difference = positions_array[-1] - positions_array[-2]
            predicted_position = positions_array[-1] + difference
            return predicted_position   # Zwracanie pozycji powiekszonej o róznicę w pozycjach, gdy za mało danych do aproksymacji
    
        time_indices = np.arange(len(self.positions_history))
        coeffs_x = np.polyfit(time_indices, positions_array[:, 0], 2)
        coeffs_y = np.polyfit(time_indices, positions_array[:, 1], 2)

        # Przewidywanie następnej pozycji powiększonej o licznik zgubionych pozycji
        next_time_index = time_indices[-1] + self.frames_since_seen
        predicted_x = np.polyval(coeffs_x, next_time_index)
        predicted_y = np.polyval(coeffs_y, next_time_index)

        return (predicted_x, predicted_y, *self.position[2:])

    def increment_frames_since_seen(self):
        self.frames_since_seen += 1
