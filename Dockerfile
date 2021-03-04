FROM python:3.8-buster

RUN mkdir -p /app/part_feeder
ADD requirements.txt /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r /app/requirements.txt \
    && apt-get purge -y --auto-remove gcc

ADD part_feeder /app/part_feeder

WORKDIR /app
ENV FLASK_APP="part_feeder"
ENTRYPOINT [ "python", "-m", "flask", "run", "--host", "0.0.0.0", "--port", "80" ]
