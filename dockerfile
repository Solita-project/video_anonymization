FROM python; 3.13

WORKDIR /VIDEO_ANONYMIZATION

COPY requirements.txt

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
