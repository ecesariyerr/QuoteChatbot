FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your bridge code and the rag_api package
COPY app.py .
COPY rag_api ./rag_api

EXPOSE 5055

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5055"]

