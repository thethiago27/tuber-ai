FROM python:3.9


WORKDIR /app
ADD . /app

COPY . .

RUN apt-get update

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]

