# Use an official Python runtime as a parent image
FROM python:3.9-slim

#System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download necessary NLP models for Python, if your application requires them
# Ensure to replace 'your_model' with the specific models you use
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download en_core_web_sm
# Add other models as necessary

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Copy the current directory contents into the container at /app
COPY . /app

#non-root user: security to limit priveliges
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH


WORKDIR $HOME/app

COPY --chown=user . $HOME/app

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app.py when the container launches
CMD ["streamlit", "run", "Tool.py", "--server.port=8501", "--server.address=0.0.0.0"]



