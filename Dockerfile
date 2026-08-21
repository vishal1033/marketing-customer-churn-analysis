FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8001

# Note: app/model/*.joblib must exist (run src/train.py locally first).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
