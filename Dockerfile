FROM python:latest

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .  

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application after installing dependencies
COPY app/ .  

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
