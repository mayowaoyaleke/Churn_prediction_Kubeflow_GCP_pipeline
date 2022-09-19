FROM python: 3.8

COPY Requirements.txt .

RUN pip install -r Requirements.txt