FROM python:3.11-slim

WORKDIR /app

# Copy file
COPY project/traffic_detection/templates/traffic.html ./templates/traffic.html
COPY project/traffic_detection/app.py app.py
COPY requirement.txt requirement.txt
COPY project/traffic_detection/./weights ./weights

# Create folder static/uploads 
RUN mkdir -p static/uploads

# Install dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]