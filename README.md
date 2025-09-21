## knowledge-bot-telegram

Телеграм-бот с RAG для ответа на вопросы о бытовой технике.

## Запуск

### Подготовка данных (`data/`)

Положите PDF-файлы:

- `data/daichi.pdf`
- `data/dantex.pdf`

### Файл окружения (`.env`)

Укажите переменные (пример расположен в `.env.example`):

```.env
TELEGRAM_BOT_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GPTUNNEL_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
QDRANT_URL=http://localhost:6333
```

### Шаг 1: Запуск Qdrant

```
docker compose up -d qdrant
```

### Шаг 2: Инициализация коллекции

Создайте виртуальное окружение (с помощью [rye](https://rye.astral.sh/)) и запустите скрипт инициализации базы данных:

```
# избегаем установки pytorch с CUDA (https://github.com/astral-sh/rye/issues/1210)
UV_INDEX_STRATEGY="unsafe-best-match" rye sync
rye run python devtools/seed_database.py
```

Примечание: скрипт пропускает инициализацию, если коллекция уже создана.

### Шаг 3: Запуск бота в Docker

```
docker compose up -d --build bot
```

Логи бота:

```
docker compose logs -f bot
```
