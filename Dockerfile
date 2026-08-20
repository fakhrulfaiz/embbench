FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir .

ENV HF_HOME=/mnt/c/ml-cache/huggingface
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "embbench.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
