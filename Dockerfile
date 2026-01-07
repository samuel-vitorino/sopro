FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --upgrade pip \
 && pip install -r demo/requirements.txt . 

EXPOSE 8000

CMD ["uvicorn", "server:app", "--app-dir", "demo", "--host", "0.0.0.0", "--port", "8000"]
