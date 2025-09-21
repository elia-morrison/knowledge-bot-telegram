## knowledge-bot-telegram

Телеграм-бот с RAG для ответа на вопросы о бытовой технике.

## Запуск

### Подготовка данных (`data/`)

Положите PDF файлы:

- `data/daichi.pdf`
- `data/dantex.pdf`

### Файл окружения (`.env` в корне, смотри `.env.example`)

Укажите переменные:

```
TELEGRAM_BOT_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GPTUNNEL_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
QDRANT_URL=http://localhost:6333
```

### Шаг 1: запуск Qdrant

```
docker compose up -d qdrant
```

### Шаг 2: инициализация коллекции

Создайте виртуальное окружение (с помощью [rye](https://rye.astral.sh/)) и запустите скрипт инициализации базы данных:

```
rye sync
rye run python devtools/seed_database.py
```

Примечание: скрипт пропускает инициализацию, если коллекция уже создана.

### Шаг 3: запустить бота в Docker

```
docker compose up -d --build bot
```

Логи бота:

```
docker compose logs -f bot
```
