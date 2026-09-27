FROM python:3.11-slim
COPY requirements.txt /opt/hermes/requirements.txt
RUN pip install --no-cache-dir -r /opt/hermes/requirements.txt
COPY runner.py /opt/hermes/runner.py
USER 65534:65534
ENTRYPOINT ["python", "-I", "/opt/hermes/runner.py"]
