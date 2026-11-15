FROM apache/airflow:3.1.3-python3.11

# System libs needed by OpenCV (pulled in by DeepFace)
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/airflow/deepface_home && chown -R airflow:0 /opt/airflow/deepface_home

USER airflow
COPY requirements.txt /requirements.txt
# Pin apache-airflow so pip never upgrades/downgrades it while resolving our deps
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
