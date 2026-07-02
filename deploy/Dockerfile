# Single image for every deployable unit; the ROLE env var selects the service.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Install deps first for layer caching.
COPY pyproject.toml requirements.txt requirements-service.txt ./
COPY src ./src
RUN pip install -r requirements.txt -r requirements-service.txt && pip install -e .

# The MCP role needs the dataset for its dataset_sample tool. data.csv is the
# challenge file and is NOT committed to git; place it at the repo root before
# building.
COPY data.csv ./data.csv

ENV PORT=8080
EXPOSE 8080
CMD ["python", "-m", "truthfulness_service.run"]
