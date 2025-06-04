import json
from typing import Dict, List, Tuple, Optional
import os
import random
from beat_this.inference import File2Beats
from beat_this.inference import File2File
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import correlate
from typing import List, Dict, Tuple
from dtaidistance import dtw
class HeartBeatRecommender:
    """
    Класс для управления коллекцией песен с временными метками битов.
    
    Хранит песни в формате:
    {
        "название песни": [время_бита1, время_бита2, ...],
        ...
    }
    """

    def __init__(self):
        """
        Инициализация менеджера.
        
        Args:
            beat_detector: Объект для детекции битов в аудио (опционально)
        """
        self.songs: Dict[str, List[float]] = {}
        self.beat_detector = File2Beats(checkpoint_path="final0", dbn=False)
    
    def add_song(self, song_name: str, beat_times: List[float]) -> None:
        """
        Добавляет или обновляет песню в коллекции.
        
        Args:
            song_name: Название песни
            beat_times: Список временных меток битов в секундах
            
        Raises:
            ValueError: Если данные битов имеют неверный формат
        """
        if not isinstance(beat_times, list) or not all(
            isinstance(t, (int, float)) for t in beat_times
        ):
            raise ValueError("Биты должны быть списком чисел (float/int)")
        
        self.songs[song_name] = beat_times
    
    def add_song_from_audio(self, audio_path: str) -> None:
        """
        Добавляет песню, анализируя биты из аудиофайла.
        
        Args:
            audio_path: Путь к аудиофайлу
            
        Raises:
            ValueError: Если детектор битов не установлен
            FileNotFoundError: Если файл не найден
        """
        if self.beat_detector is None:
            raise ValueError("Детектор битов не установлен. Используйте set_beat_detector()")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")
        
        try:
            # Получаем название песни из имени файла
            song_name = os.path.splitext(os.path.basename(audio_path))[0]
            
            # Получаем времена битов (используем только первый элемент кортежа)
            _, downbeats = self.beat_detector(audio_path)
            #print(f"downbeats {downbeats}, type {type(downbeats)}")
            # Добавляем песню в коллекцию
            self.add_song(song_name, downbeats.tolist())
            
        except Exception as e:
            raise ValueError(f"Ошибка анализа аудио: {e}")

    def add_song_from_file(self, file_path: str) -> None:
        """
        Добавляет песню, загружая времена битов из текстового файла.
        
        Args:
            file_path: Путь к файлу с временами битов (по одному в строке)
            
        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если файл имеет неверный формат
        """
        beat_times = []
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            beat_times.append(float(line))
                        except ValueError:
                            raise ValueError(
                                f"Неверный формат в строке {line_num}: '{line}'"
                            )
            
            song_name = os.path.splitext(os.path.basename(file_path))[0]
            self.add_song(song_name, beat_times)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл не найден: {file_path}")

    def remove_song(self, song_name: str) -> bool:
        """
        Удаляет песню из коллекции.
        
        Args:
            song_name: Название песни для удаления
            
        Returns:
            True если песня была удалена, False если не найдена
        """
        if song_name in self.songs:
            del self.songs[song_name]
            return True
        return False
    
    def get_song(self, song_name: str) -> Optional[List[float]]:
        """
        Возвращает времена битов для указанной песни.
        
        Args:
            song_name: Название песни
            
        Returns:
            Список времен битов или None, если песня не найдена
        """
        return self.songs.get(song_name)
    
    def find_best_matches(self, 
                        heartbeat_downbeats: List[float], 
                        top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Находит топ-N песен, наиболее подходящих под ритм сердцебиения.
        
        Args:
            heartbeat_downbeats: Времена сильных долей сердцебиения
            top_n: Количество лучших совпадений
            
        Returns:
            Список кортежей (название, оценка совпадения 0-1)
        """
        if not heartbeat_downbeats:
            raise ValueError("Нужно хотя бы 2 даунбита сердцебиения")
            
        if len(heartbeat_downbeats) < 2:
            raise ValueError("Нужно хотя бы 2 даунбита сердцебиения")
        
        # Нормализуем интервалы сердцебиения
        hb_intervals = np.diff(heartbeat_downbeats)
        hb_norm = hb_intervals / np.mean(hb_intervals)
        
        results = []
        
        for name, song_downbeats in self.songs.items():
            if len(song_downbeats) < 2:
                continue
                
            # Нормализуем интервалы песни
            song_intervals = np.diff(song_downbeats)
            song_norm = song_intervals / np.mean(song_intervals)
            
            # Вычисляем минимальную длину для сравнения
            min_len = min(len(hb_norm), len(song_norm))
            
            # Сравниваем паттерны интервалов
            pattern_diff = np.mean(np.abs(
                hb_norm[:min_len] - song_norm[:min_len]
            ))
            
            # Оценка совпадения (чем ближе к 0 разница - тем лучше)
            score = 1 / (1 + pattern_diff)
            results.append((name, score))
        
        # Сортируем и возвращаем топ-N
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]
    
    def find_best_matches_with_dynamic_time_warping(self,
                                                  heartbeat_downbeats: List[float],
                                                  top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Более точное сравнение с Dynamic Time Warping.
        """
        
        
        if len(heartbeat_downbeats) < 2:
            raise ValueError("Нужно хотя бы 2 даунбита сердцебиения")
            
        # Преобразуем в numpy array
        hb_array = np.array(heartbeat_downbeats)
        
        results = []
        
        for name, song_downbeats in self.songs.items():
            if len(song_downbeats) < 2:
                continue
                
            song_array = np.array(song_downbeats)
            
            # Нормализуем по длительности
            hb_norm = (hb_array - hb_array[0]) / (hb_array[-1] - hb_array[0])
            song_norm = (song_array - song_array[0]) / (song_array[-1] - song_array[0])
            
            # Вычисляем DTW расстояние
            distance = dtw.distance(hb_norm, song_norm)
            
            # Преобразуем в оценку (чем меньше расстояние - тем лучше)
            score = 1 / (1 + distance)
            results.append((name, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]
    
    def save_to_json(self, file_path: str) -> None:
        """
        Сохраняет все песни в JSON-файл.
        
        Args:
            file_path: Путь для сохранения файла
            
        Raises:
            IOError: При ошибках записи файла
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({"songs": self.songs}, f, indent=4, ensure_ascii=False)
        except Exception as e:
            raise IOError(f"Ошибка записи JSON: {e}")
    
    def load_from_json(self, file_path: str) -> None:
        """
        Загружает песни из JSON-файла (заменяет текущую коллекцию).
        
        Args:
            file_path: Путь к JSON-файлу
            
        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если файл имеет неверный формат
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data.get("songs", {}), dict):
                raise ValueError("Неверный формат данных в JSON")
            
            self.songs = {
                song: [float(t) for t in times] 
                for song, times in data["songs"].items()
            }
            
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON-файл не найден: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Файл {file_path} не является валидным JSON")
        except Exception as e:
            raise ValueError(f"Ошибка загрузки JSON: {e}")
    
    def clear_all(self) -> None:
        """Очищает коллекцию песен."""
        self.songs.clear()
        
        
def generate_heartbeat_timestamps(bpm: int, duration_seconds: int, variability: float = 0.1) -> List[float]:
    """
    Генерирует реалистичные временные метки сердцебиения с небольшими вариациями
    
    Аргументы:
        bpm: Ударов в минуту (целевая частота сердцебиения)
        duration_seconds: Длительность генерации в секундах
        variability: Величина случайных вариаций (0.1 = 10% вариаций)
    
    Возвращает:
        Список временных меток (в секундах) для каждого удара сердца
    """
    if bpm <= 0 or duration_seconds <= 0:
        return []
    
    average_interval = 60.0 / bpm  # Average time between beats in seconds
    timestamps = []
    current_time = 0.0
    
    while current_time <= duration_seconds:
        # Add some natural variability to the heartbeat
        variation = random.uniform(1 - variability, 1 + variability)
        current_interval = average_interval * variation
        current_time += current_interval
        
        if current_time <= duration_seconds:
            timestamps.append(round(current_time, 3))  # Round to milliseconds
    
    return timestamps

# Пример использования
if __name__ == "__main__":
    # Создаем менеджер
    manager = HeartBeatRecommender()
    try:
        manager.add_song_from_audio("nabrosok.wav")
        print("Новая песня добавлена")
    except Exception as e:
        print(f"Ошибка в добавлении песни {e}")
        print(e)
    # Добавляем песню с готовыми данными
    # Создаем менеджер и добавляем песни

    manager.add_song("Song 1", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])  # Четкий 4/4
    manager.add_song("Song 2", [1.0, 1.3, 1.6, 2.0, 2.3, 2.6, 3.0, 3.3])  # Тройной размер
    manager.add_song("Song 3", [1.0, 1.8, 2.6, 3.4, 4.2, 5.0])  # Нестандартный ритм

    heartbeat = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]  # Стандартный 4/4
    print(f"Базовый ритм сердца: {heartbeat}")
    # Находим лучшие совпадения
    matches = manager.find_best_matches_with_dynamic_time_warping(heartbeat)
    print("Лучшие совпадения:")
    for name, score in matches:
        print(f"- {name}: {score:.2f}")
        
    
    try:
        hbpm = 70
        heartbeat_gen = generate_heartbeat_timestamps(hbpm, 60)
        print(f"Сгенерирован ритм сердца {hbpm} ударов в минуту: {heartbeat_gen}")
        # Биты сердцебиения (пример)
        matches = manager.find_best_matches_with_dynamic_time_warping(heartbeat_gen)
        print("Лучшие совпадения:")
        for name, score in matches:
            print(f"- {name}: {score:.2f}")
    except:
        print("Ошибка в поиске песни")
   
    
        
        