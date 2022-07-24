FROM python:3.7

# Create app directory
WORKDIR /app

# Copy all files to image
COPY . /app

# Update pip
RUN python -m pip install --upgrade pip

# Install requirements
RUN pip install -r requirements.txt

# Expose app
EXPOSE 5000

# Execute init
CMD ["python", "app.py"]