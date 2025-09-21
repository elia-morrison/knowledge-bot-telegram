FROM python:3.12.6-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.lock /app/requirements.lock
COPY ./app.py ./pyproject.toml ./README.md /app


RUN pip config set global.extra-index-url https://download.pytorch.org/whl/cpu

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.lock

COPY ./src /app/src
COPY ./data /app/data

CMD ["python", "app.py"]


