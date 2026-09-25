FROM python:3.11-slim
RUN pip install --no-cache-dir pytest==8.3.5
COPY runner.py /opt/hermes/runner.py
USER 65534:65534
ENTRYPOINT ["python", "-I", "/opt/hermes/runner.py"]
