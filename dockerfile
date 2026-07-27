FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install wget (and bash) required by your main.sh script
RUN apt-get update && apt-get install -y --no-install-recommends wget bash && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Grant execute permissions to the launcher script
RUN chmod +x main.sh

EXPOSE 5000

# Execute the bash script instead of gunicorn directly
CMD ["./main.sh"]