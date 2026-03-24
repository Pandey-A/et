FROM python:3.10

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application
COPY . .

# Flask environment variables
ENV FLASK_APP=app3.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose port 5000
EXPOSE 5050

# Run Flask
CMD ["flask", "run"]
