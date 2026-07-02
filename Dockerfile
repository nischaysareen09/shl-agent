FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


EXPOSE 8000

# Render/Fly/Railway inject $PORT; default to 8000 for local docker run
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
