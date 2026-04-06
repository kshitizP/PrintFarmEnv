---
title: PrintFarmEnv
emoji: 🖨️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# PrintFarmEnv

A hardware DevOps environment for managing a fleet of 3D printers and inventory. Built for OpenEnv hackathon.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can test the environment and see a baseline agent using the `inference.py` script:
```bash
python inference.py
```

## Repository Structure

- `printfarm_env/`: Contains the core environment logic including the API simulator and task handlers.
- `openenv.yaml`: OpenEnv configuration file defining tasks and dependencies.
- `inference.py`: Sample minimal agent implementation.
