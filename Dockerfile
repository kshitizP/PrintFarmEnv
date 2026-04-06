FROM python:3.10-slim

# Create a non-root user (good practice for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app
COPY --chown=user . /app/

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install openenv-core openai pydantic

# The validation script expects the environment to be importable and responsive
CMD ["python", "inference.py"]
