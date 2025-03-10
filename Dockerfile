FROM python:latest

WORKDIR /app

# copy and install all the requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


#copies all the files in app folder to working dir
COPY app/ . 

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000","--reload"] 

