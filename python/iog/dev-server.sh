#!/usr/bin/env bash

../../.venv/bin/uvicorn server:app --reload --host 0.0.0.0 --port 5032 --log-level info --env-file ../../.env
