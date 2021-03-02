FROM tiangolo/meinheld-gunicorn-flask:python3.8-2020-12-19

ADD requirements.txt /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r /app/requirements.txt \
    && apt-get purge -y --auto-remove gcc

RUN mkdir -p /app/part_feeder
ADD part_feeder /app/part_feeder
ENV APP_MODULE="part_feeder:create_app()"
ENV WEB_CONCURRENCY="2"
