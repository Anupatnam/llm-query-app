FROM mcr.microsoft.com/devcontainers/python:3.11

WORKDIR /workspace

# Pre-installed: python, pip, venv, system tools

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/main.py"]


