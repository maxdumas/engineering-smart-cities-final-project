FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/universal

# Copy environment.yml  to a temp location so we update the environment.
COPY environment.yml /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp
