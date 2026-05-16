FROM python:3.10-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY backend/ /app/backend/
COPY ai/ /app/ai/
COPY shared/ /app/shared/

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "warning"]
