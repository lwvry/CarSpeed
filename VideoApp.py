import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from threading import Thread
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from VideoProcessor import VideoProcessor
import time
from queue import Queue
import numpy as np

# Słownik modeli dronów
DRONES = {
    "DJI mini 4 pro": "DJI mini 4 pro",
    "DJI air 2s": "DJI air 2s"
}


class VideoApp:
    """
    Klasa aplikacji
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Car speed")
        self.root.attributes("-fullscreen", True)   # Pełny ekran
        self.is_processing = False  # Flaga przetwarzania nagrania
        self.is_refreshing_graph = False    # Flaga odświeżania wykresu
        self.video_path = None  # Ścieżka do nagrania
        self.output_path = None # Ścieżka do zapisu przetworzonego nagrania
        self.video_processor = None # Obiekt do przetwarzania klatek nagrania 
        self.frame_queue = Queue(maxsize=2) # Kolejka klatek do wyświetlania
        self.fps = 0    # Wskaznik fps
        self.prev_time = time.time()
        self.output_writer = None   # Obiekt zapisu filmu
        self.start_coordiantes = None   # Wysokość startowa drona

        # Konfiguracja motywu dla wykresu
        sns.set_theme(style="darkgrid")

        # Ramka dla ustawień
        settings_frame = ttk.LabelFrame(root, text="Settings", padding=(5, 10))
        settings_frame.grid(row=0, column=0, rowspan=2, sticky="nw", padx=5, pady=10)

        # UI: Wybór ścieżki wideo
        ttk.Label(settings_frame, text="Video file path:").grid(row=0, column=0, sticky="w", pady=5)
        self.video_path_var = tk.StringVar()
        self.video_entry = ttk.Entry(settings_frame, textvariable=self.video_path_var, width=20)
        self.video_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="Select video", command=self.select_video).grid(row=0, column=2, padx=5)

        # UI: Wybór ścieżki zapisu wideo
        ttk.Label(settings_frame, text="Save video to:").grid(row=1, column=0, sticky="w", pady=5)
        self.output_path_var = tk.StringVar()
        self.output_entry = ttk.Entry(settings_frame, textvariable=self.output_path_var, width=20)
        self.output_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="Select path", command=self.select_output_path).grid(row=1, column=2, padx=5)

        # UI: Wybór modelu drona
        ttk.Label(settings_frame, text="Select drone model:").grid(row=2, column=0, sticky="w", pady=5)
        self.drone_model_var = tk.StringVar(value=list(DRONES.keys())[0])
        self.drone_model_menu = ttk.OptionMenu(settings_frame, self.drone_model_var, list(DRONES.keys())[0], *DRONES.keys())
        self.drone_model_menu.grid(row=2, column=1, padx=5, pady=5)

        # UI: Wybór ID pojazdu
        ttk.Label(settings_frame, text="Car ID:").grid(row=3, column=0, sticky="w", pady=5)
        self.selected_car_id = tk.StringVar(value="0")
        self.car_id_spinbox = ttk.Spinbox(settings_frame, from_=0, to=1000, width=20, textvariable=self.selected_car_id, wrap=True)
        self.car_id_spinbox.grid(row=3, column=1, padx=5, pady=5)

        # UI: Wybór wysokości startowej
        ttk.Label(settings_frame, text="Start altitude:").grid(row=4, column=0, sticky="w", pady=5)
        self.start_coordiantes = tk.StringVar()
        self.start_coordiantes_entry = ttk.Entry(settings_frame, textvariable=self.start_coordiantes, width=20)
        self.start_coordiantes_entry.grid(row=4, column=1, padx=5, pady=5)

        # Panel kontrolny
        control_frame = ttk.Frame(settings_frame, padding=(10, 10))
        control_frame.grid(row=5, column=0, columnspan=3, pady=10)
        self.start_btn = ttk.Button(control_frame, text="Start", command=self.start_processing, state=tk.DISABLED)
        self.start_btn.grid(row=0, column=0, padx=5)
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Show chart", command=self.show_speed_graph).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Exit", command=self.quit_app).grid(row=0, column=3, padx=5)

        # UI: Pasek ładowania
        ttk.Label(settings_frame, text="Processing Progress:").grid(row=6, column=0, sticky="w", pady=5)
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(settings_frame, orient="horizontal", length=300, mode="determinate", variable=self.progress_var)
        self.progress_bar.grid(row=6, column=1, columnspan=2, pady=5)
        self.progress_label = ttk.Label(settings_frame, text="0%")
        self.progress_label.grid(row=7, column=1, columnspan=2, pady=5)

        # Wskaźnik fps
        self.fps_label = ttk.Label(settings_frame, text="FPS: 0")
        self.fps_label.grid(row=8, column=0, columnspan=3, pady=5)


        # Canvas na do wyświetlania nagrania
        self.canvas = tk.Canvas(root, bg="black")
        self.canvas.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=0, pady=0)

        # Ramka dla wykresu
        graph_frame = ttk.LabelFrame(root, text="Velocity chart", padding=(10, 10))
        graph_frame.grid(row=2, column=0, sticky="nw", padx=5, pady=5)

        self.figure = Figure(figsize=(5, 4), dpi=90)
        self.ax = self.figure.add_subplot()
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # Dodanie wykresu do interfejsu
        self.graph_canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Ustawienie proporcji siatki
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=1)

    def get_start_altitude(self):
        value = self.start_coordiantes.get().strip()
        if not value:
            return None
        try:
            value = float(value)
            return value
        except ValueError:
            pass
        messagebox.showerror("Error", "Invalid float value")
        return None
    def select_video(self):
        """
        Otwiernie okna do wyboru nagrania
        """
        self.video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
        )
        if self.video_path:
            self.video_path_var.set(self.video_path)
            self.start_btn.config(state=tk.NORMAL)

    def select_output_path(self):
        """
        Otwieranie okna do wyboru ściezki zapisu
        """
        self.output_path = filedialog.asksaveasfilename(
            title="Select Video Save Path",
            defaultextension=".mp4",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
        )
        if self.output_path:
            self.output_path_var.set(self.output_path)
            
    def update_progress_bar(self, current, total):
        """
        Aktualizacja paska postępu
        """ 
        progress = int((current / total) * 100)
        self.progress_var.set(progress)
        self.progress_label.config(text=f"{progress}%")
        self.root.update_idletasks()

    def load_video_processor(self):
        """
        Inicjalizacja klasy VideoProcessor i ewentualnej klasy VideoWriter
        """
        if not self.video_path:
            messagebox.showerror("Error", "No video file selected")
            return False
        try:
            selected_drone = self.drone_model_var.get()
            altitude = self.get_start_altitude()
            self.video_processor = VideoProcessor(  # Inicjalizacja VideoProcessor
                self.video_path, selected_drone,
                altitude,
                model_path="models/drone7liten-obb-dota_and_data22.pt"   
            )
            if self.output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.output_writer = cv2.VideoWriter(
                    self.output_path, 
                    fourcc, 
                    self.video_processor.fps, 
                    (self.video_processor.frame_width, self.video_processor.frame_height)
                )
            else:
                self.output_writer = None   # Brak zapisu nagranie, kiedy nie podano ściezki zapisu
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video processor: {e}")
            return False

    def start_processing(self):
        """
        Przygotowanie do rozpoczęcia procesu przetwarzania nagrania
        """
        if not self.load_video_processor():
            return
        self.total_frames = self.video_processor.get_total_frame_count()
        self.frame_count = 0
        self.is_processing = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        Thread(target=self.process_video, daemon=True).start()  # Utworzenie wątku do przetwarzania nagrania
        self.update_canvas()    # Wywołanie rekurencyjnej funkcji do wyświetlania klatek
        self.update_fps_label() # Wywołanie rekurencyjnej funkcji do wskaznika fps

    def process_video(self):
        """
        Przetwarzanie nagrania
        """
        try:
            while self.is_processing:   # Flaga przetwarzania nagrania
                frame, is_frame_available = self.video_processor.process_frame()    # Przetworzenie kolejnej klatki nagrania
                if is_frame_available:  #Sprawdzenie czy jest to koniec nagrania lub uszkodzone nagranie
                    if self.output_writer:
                        self.output_writer.write(frame) # Ewentualny zapis do pliku
                    frame = self.scale_frame_for_display(frame) # Przeskalowanie klatki
                    if not self.frame_queue.full(): 
                        self.frame_queue.put(frame) # Wysłanie klatki do kolejki
                    self.frame_count += 1

                    self.update_progress_bar(self.frame_count, self.total_frames)
                    current_time = time.time()
                    fps = 1.0 / (current_time - self.prev_time)
                    self.prev_time = current_time
                    self.fps = fps
                else:
                    break
        finally:
            self.is_processing = False
            if self.output_writer:
                self.output_writer.release() # Zwolnienie klasy do nagrywania
            self.video_processor.release()  # Zwolnienie danych w VideoProcessor
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def update_fps_label(self):
        """
        Aktualizacja wskaznika fps
        """
        self.fps_label.config(text=f"FPS: {self.fps:.2f}")
        if self.is_processing:
            self.root.after(1000, self.update_fps_label)

    def scale_frame_for_display(self, frame):
            """
            Skalowanie klatki do wymiarów okna
            """
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width == 0 or canvas_height == 0:
                return frame
            frame_height, frame_width = frame.shape[:2]
            scale = min(canvas_width / frame_width, canvas_height / frame_height)   # Obliczanie współczynnika skalowania
            new_width = int(frame_width * scale)    # Nowa szerokość klatki po skalowaniu
            new_height = int(frame_height * scale)  # Nowa wysokość klatki po skalowaniu
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            #resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            delta_w = canvas_width - new_width
            delta_h = canvas_height - new_height
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            # Dodanie czarnych pasków na dole i na górze klatki
            return cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    def update_canvas(self):
        """
        Wyswietlanie przetworzonych klatek nagrania
        """
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            photo = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = ImageTk.PhotoImage(image=Image.fromarray(photo))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.frame_image = img
        if self.is_processing:
            self.root.after(33, self.update_canvas) # Odświeżanie co 33 ms
   

    def show_speed_graph(self):
        """
        Inicjalizacja wykresu prędkości dla podanego ID
        """
        self.stop_graph_refresh()
        car_id = self.selected_car_id.get()
        if not car_id:
            messagebox.showerror("Error", "Enter your vehicle ID")
            return
        try:
            car_id_int = int(car_id)
            self.current_car_id = car_id_int
            self.is_refreshing_graph = True
            self.refresh_speed_graph(car_id_int)
        except ValueError:
            messagebox.showerror("Error", "Vehicle ID must be a number")
        except Exception as e:
            messagebox.showerror("Error", f"Chart failed to display: {e}")

    def refresh_speed_graph(self, car_id):
        """
        Wyświetlanie wykresu prędkości
        """
        if not self.is_refreshing_graph or self.current_car_id != car_id:
            return
        try:
            speed_data = self.video_processor.get_speed_history(car_id)
            if not speed_data:
                self.ax.clear()
                self.ax.set_title(f"No data available for vehicle ID {car_id}")
            else:
                avg_speed = np.mean(speed_data)
                
                timestamps = [i for i in range(len(speed_data))]
                self.ax.clear()
                sns.lineplot(
                    x=timestamps,
                    y=speed_data,
                    ax=self.ax,
                    label=f"Vehicle ID {car_id}",
                    color="tab:blue",
                    linewidth=2.5
                )
                
                self.ax.axhline(y=avg_speed, color='red', linestyle='--', label=f"Avg Speed: {int(avg_speed)} km/h")
                self.ax.set_ylabel("Velocity (km/h)", fontsize=12)
                self.ax.set_title("Speed Graph", fontsize=14, fontweight="bold")
                self.ax.set_ylim(0, 100)
                self.ax.legend(fontsize=10, loc="upper right", frameon=True)
                self.ax.set_xticklabels([])
            self.graph_canvas.draw()
        except Exception as e:
            self.ax.clear()
            self.ax.set_title(f"Error: {e}")
            self.graph_canvas.draw()
        self.root.after(1000, lambda: self.refresh_speed_graph(car_id)) # Odświeżanie wykresu co sekundę (rekurencja)

    def stop_processing(self):
        """
        Zatrzymanie przetwarzania nagrania
        """
        self.is_processing = False
        self.stop_graph_refresh()

    def stop_graph_refresh(self):
        """
        Zatrzymanie odświeżania wykresu
        """
        self.is_refreshing_graph = False
        self.current_car_id = None

    def quit_app(self):
        """
        Zamknięcie aplikacji
        """
        self.stop_processing()
        self.root.destroy()


