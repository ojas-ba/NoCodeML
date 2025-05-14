FROM python:3.10

# 1. Install OS‑level build deps (for PyYAML and any other C‑exts)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libyaml-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Upgrade pip/setuptools/wheel and pin Cython < 3.0
#    so that PyYAML 5.x can build cleanly
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir "cython<3.0.0"

# 3. Copy only your requirements (to leverage Docker cache)
COPY requirements.txt .  

# 4. Install Python deps WITHOUT isolation, so it reuses our pinned Cython
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

# 5. Bring in the rest of your code
COPY app/ .  

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
