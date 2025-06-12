FROM mcr.microsoft.com/devcontainers/python:3.11

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/main.py"]



