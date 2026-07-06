FROM python:3.11-slim
WORKDIR /app
# One of eyecite's dependencies ships no prebuilt wheel for Linux ARM so pip
# has to compile it. Need this line because slim strips out the compiler needed
# to compile it.
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir . datasets voyageai google-genai
CMD ["mischar", "--help"]