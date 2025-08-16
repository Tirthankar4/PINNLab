# Logging

This folder contains all logging-related files for the PINNLab application.

## Contents

- **logs/** - Application log files (automatically created)
- **README.md** - This file

## Log Files

The application automatically creates log files in the `logs/` subfolder:
- Log files are rotated when they reach 10MB
- Up to 5 backup files are kept
- Logs include timestamps, log levels, and messages

## Configuration

Logging is configured in `app.py` using the SafeLogger class:
- Log level: INFO (configurable)
- Output: Both console and file
- Format: Timestamp - Module - Level - Message
