FROM svizor/zoomcamp-model:3.9.12-slim

ENV PYTHONUNBUFFERED=TRUE

RUN pip --no-cache-dir install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --deploy --system && \
    rm -rf /root/.cache

COPY ["scoring_servise2.py", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen", "0.0.0.0:9696", "scoring_servise2:app"] 
