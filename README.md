# Документация: `HeartBeatRecommender`

## 📦 Установка

```bash
python -m venv heartrec
. heartrec/bin/activate

pip install https://github.com/CPJKU/beat_this/archive/main.zip
pip install dtaidistance
```

## 📘 Описание

Класс `HeartBeatRecommender` предназначен для управления коллекцией музыкальных треков и рекомендации песен по ритму сердцебиения. Сравнение ритмов происходит с использованием методов кросс-корреляции и DTW (Dynamic Time Warping).

## 🔧 Основные функции

### 1. Хранение данных

```python
{"название песни": [время_даунбитов]}
```

**Пример:**
```python
{
  "Rock": [1.0, 2.0, 3.0, 4.0],
  "Waltz": [1.0, 2.5, 4.0, 5.5],
  "Jazz": [1.0, 1.8, 3.2, 4.0]
}
```

### 2. Добавление песен

#### `add_song(song_name: str, downbeats: List[float])`

Добавляет песню вручную по списку даунбитов.

```python
manager.add_song("Rock", [1.0, 2.0, 3.0, 4.0])
```

### 3. Поиск лучших совпадений

#### `find_best_matches(heartbeat_downbeats: List[float], top_n: int = 3)`

Сравнение по кросс-корреляции.

```python
heartbeat = [1.0, 2.1, 3.2, 4.3]
matches = manager.find_best_matches(heartbeat)
print(matches)  # [("Rock", 0.92), ("Jazz", 0.85), ("Waltz", 0.78)]
```

#### `find_best_matches_with_dynamic_time_warping(heartbeat_downbeats, top_n=3)`

Точное сравнение с использованием DTW.

```python
matches_dtw = manager.find_best_matches_with_dynamic_time_warping(heartbeat)
print(matches_dtw)  # [("Rock", 0.95), ("Jazz", 0.88), ("Waltz", 0.80)]
```

### 4. Сохранение и загрузка данных

#### `save_to_json(file_path: str)`

Сохраняет коллекцию песен в файл.

```python
manager.save_to_json("songs.json")
```

#### `load_from_json(file_path: str)`

Загружает коллекцию песен из JSON-файла.

```python
manager.load_from_json("songs.json")
```

## 💡 Пример использования

```python
manager = HeartBeatRecommender()
manager.add_song("Rock", [1.0, 2.0, 3.0, 4.0])
manager.add_song("Jazz", [1.0, 1.8, 3.2, 4.0])

heartbeat = [1.0, 2.1, 3.2, 4.3]
matches = manager.find_best_matches(heartbeat)
print("Лучшие совпадения:", matches)
```

## 📊 Математические основы

### 1. Нормализация

I_norm = I / μ(I), где μ(I) = (tₙ - t₁)/(n - 1)

### 2. Кросс-корреляция

ρ = Σ[(x_i - μ_x)(y_i - μ_y)] / (σ_x σ_y), оценка: score = (ρ + 1)/2

### 3. Dynamic Time Warping

D(i,j) = d(i,j) + min(D(i-1,j), D(i-1,j-1), D(i,j-1))  
score = 1 / (1 + DTW_dist)

## 📁 Пример структуры файла

```json
{
  "songs": {
    "Song 1": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
    "Song 2": [1.0, 1.3, 1.6, 2.0, 2.3, 2.6, 3.0, 3.3],
    "Song 3": [1.0, 1.8, 2.6, 3.4, 4.2, 5.0]
  }
}
```

## 🧠 Применение

- Музыкальные приложения  
- Персональные фитнес-трекеры  
- Биомедицинский анализ состояния человека  

## 🚀 Возможности улучшения

- Автоматическое извлечение битов через `librosa`  
- Обучаемые модели сопоставления (ML)  
- Поддержка аудио-потоков и онлайн-обработки
