FROM python:3.12-slim

WORKDIR /app

# (scikit-learn için derleme araçları gerekebilir; wheel varsa yine de sorun değil)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# 3.12 ile uyumlu wheel'leri almak için ekstra bir şeye gerek yok; pip çözer.
# İstersen scikit-learn'ü sabitleyebilirsin: scikit-learn>=1.4
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=7000
EXPOSE 7000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000"]
