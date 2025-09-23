#!/usr/bin/env bash
# exit on error
set -e

pip install -r requirements.txt
python -m playwright install --with-deps chromium