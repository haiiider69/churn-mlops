# Step 5: containerize the FastAPI serving app.
# This builds and runs, but src/api.py still needs its TODOs filled in
# before /predict actually works.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config.yaml .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
