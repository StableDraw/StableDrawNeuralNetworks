FROM python:3.11.3-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

RUN useradd -ms /bin/bash user

USER user
WORKDIR /home/user

COPY requirements.txt requirements.txt

RUN pip install --no-warn-script-location --no-cache-dir -r requirements.txt

COPY ./app ./app

ENTRYPOINT [ "python", "-m", "propan", "run", "app.serve:app" ]