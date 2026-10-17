# Python API container: scoring + replay + genai, served by uvicorn (FastAPI).
#
# Base = IBM's LinuxONE Jupyter image. Why not python:3.12-slim?
#   pandas (and numpy, scikit-learn) publish no prebuilt s390x wheels on PyPI, so they fail to
#   install there (tested on the VM, Oct 17). IBM's image already has them compiled for s390x:
#   Python 3.12.15, pandas 3.0.6, scikit-learn 1.9.0, xgboost 3.4.1, llama_cpp_python.
# Bonus: Jupyter uses the same image, so its layers are already on disk (no extra download).
# It already runs as the non-root user "ibm-user" (uid 1000), so no user setup is needed.
FROM registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest

WORKDIR /home/ibm-user/app

# Install the extra packages first, in their own layer: Docker caches this step,
# so rebuilds after a code-only change take seconds instead of minutes.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Then copy the code (changes often, so it comes after the slow step).
COPY src/ ./src/

EXPOSE 8000
# Replaces the image's default command (which starts Jupyter) with our API server.
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
