# Base image, specific Python image version
FROM python:3.10.13-slim

# Set the working directory in the container
WORKDIR /user/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Disable the creation of virtual environment by Poetry
RUN poetry config virtualenvs.create false

# Copy only poetry files to leverage Docker cache
COPY pyproject.toml poetry.lock* ./

# Install dependencies
# --no-interaction --no-ansi
RUN poetry install --no-dev 

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
ENV APP_PORT=80

# Command to run the application
CMD uvicorn app:app --host 0.0.0.0 --port $APP_PORT --reload