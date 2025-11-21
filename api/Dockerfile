FROM python:3.10-slim
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies (increase timeout for large packages)
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy the rest of the files
COPY . /app

# Set environment variable for data directory
ENV INTRANET_DATA_DIR=/data
RUN mkdir -p /data

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
