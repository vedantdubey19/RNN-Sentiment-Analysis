FROM python:3.9-slim

WORKDIR /code

# Copy and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application files
COPY . .

# Run application using Gunicorn on default Hugging Face Spaces port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
