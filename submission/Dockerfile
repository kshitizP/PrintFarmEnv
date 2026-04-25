FROM python:3.11.12-slim-bookworm

# Create a non-root user (required for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app
COPY --chown=user . /app/

RUN pip install --no-cache-dir -r requirements.txt

# HF Spaces port
CMD ["uvicorn", "submission.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
