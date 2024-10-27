ARG IMAGE_VARIANT=bookworm
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-${IMAGE_VARIANT} AS py3

RUN wget http://www.mirbsd.org/~tg/Debs/sources.txt/wtf-bookworm.sources
RUN mkdir -p /etc/apt/sources.list.d
RUN mv wtf-bookworm.sources /etc/apt/sources.list.d/

RUN apt-get update && apt-get install --no-install-recommends --yes \
    tesseract-ocr openjdk-8-jdk

USER root
WORKDIR /app
COPY poetry.lock pyproject.toml /app/
RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install -n --no-ansi

ENTRYPOINT ["top", "-b"]
