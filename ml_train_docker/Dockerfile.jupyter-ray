FROM quay.io/jupyter/pytorch-notebook:latest

USER ${NB_UID}

# Install Lightning
RUN pip install --pre --no-cache-dir lightning && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Install Ray
RUN pip install --pre --no-cache-dir ray && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"