#!/bin/bash
exec gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 8 --timeout 0 app:app