#!/bin/bash
# Start the Flask application using Gunicorn

# Activate virtual environment (if using one)
# Uncomment and adjust the path if you have a virtual environment
# source /path/to/venv/bin/activate

# Run Gunicorn
exec gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 8 --timeout 0 app:app