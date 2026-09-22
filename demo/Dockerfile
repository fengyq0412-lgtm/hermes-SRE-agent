FROM python:3.11-slim

WORKDIR /app

COPY demo_service.py /app/demo_service.py
COPY runtime /app/runtime

RUN mkdir -p /app/logs

EXPOSE 8000

CMD ["python", "demo_service.py"]
