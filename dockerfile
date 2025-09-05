# 1) Base Miniconda
FROM continuumio/miniconda3:latest

# 2) Configurer conda
RUN conda config --add channels nvidia \
 && conda config --add channels conda-forge \
 && conda config --set channel_priority flexible

# 3) Installer dépendances système pour FSL, ROBEX, Singularity, etc.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        bc \
        tcsh \
        unzip \
        libgomp1 \
        rsync \
        squashfs-tools \
        libc-bin \
    && rm -rf /var/lib/apt/lists/*

# 4) Copier et installer vos outils externes
#    On présume que /tools contient trois dossiers :
#      - fsl509/      (FSL 5.0.9)
#      - ROBEX/       (robust brain extraction)
#      - singul/      (vos scripts Singularity ou utilitaires)
COPY tools /opt/tools/

# 4.a) FSL
ENV FSLDIR=/opt/tools/fsl509
ENV PATH=$FSLDIR/bin:$PATH
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV FSLDISPLAYUSENEW=true

RUN ln -s $FSLDIR/bin/* /usr/local/bin/ \
 && echo "source $FSLDIR/etc/fslconf/fsl.sh" >> /root/.bashrc

# 4.b) ROBEX
ENV ROBEXDIR=/opt/tools/ROBEX
RUN ln -s $ROBEXDIR/ROBEX_auto /usr/local/bin/ROBEX_auto

# 4.c) singul (script(s))
ENV SINGULDIR=/opt/tools/singul
RUN ln -s $SINGULDIR/* /usr/local/bin/
RUN mkdir -p /usr/local/etc/singularity  
COPY tools/singul/etc/singularity/singularity.conf /usr/local/etc/singularity/singularity.conf  
COPY tools/singul/etc/singularity/nvliblist.conf /usr/local/etc/singularity/nvliblist.conf
ENV SINGULARITY_CONF=/usr/local/etc/singularity/singularity.conf
RUN ln -s /usr/sbin/ldconfig /usr/sbin/ldconfig.real
RUN mkdir -p /usr/local/libexec/singularity/bin
RUN ln -s /opt/tools/singul/etc/libexec/singularity/bin/starter /usr/local/libexec/singularity/bin/starter
RUN mkdir -p /usr/local/var/singularity/mnt/session


# 5) Créer tous les environnements conda pour vos algos
#    On part du principe que vous avez :
#      algos/DISARMpp/disarmpp_env.yml
#      algos/HACA3/haca3_env.yml
#      algos/IGUANE/iguane_env.yml
#      algos/MURD/murd_env.yml
#      algos/STGAN/stgan_env.yml
COPY algos /opt/toolkit/algos
WORKDIR /opt/toolkit/algos
RUN chmod +x /opt/toolkit/algos/DISARMpp/preprocessing/mri_prep.sh

RUN conda env create -f DISARMpp/disarmpp_env.yml && \
    conda clean --all --yes && \
    rm -rf /opt/conda/pkgs/* && \
    rm -rf /root/.cache/pip && \
    rm -rf /opt/conda/envs/disarmpp_env/conda-meta \
    /opt/conda/envs/disarmpp_env/include \
    /opt/conda/envs/disarmpp_env/share

RUN conda env create -f HACA3/haca3_env.yml && \
    conda clean --all --yes && \
    rm -rf /opt/conda/pkgs/* && \
    rm -rf /root/.cache/pip && \
    rm -rf /opt/conda/envs/haca3_env/conda-meta \
    /opt/conda/envs/haca3_env/include \
    /opt/conda/envs/haca3_env/share

RUN conda env create -f IGUANE/iguane_env.yml && \
    conda clean --all --yes && \
    rm -rf /opt/conda/pkgs/* && \
    rm -rf /root/.cache/pip && \
    rm -rf /opt/conda/envs/iguane_env/conda-meta \
    /opt/conda/envs/iguane_env/include \
    /opt/conda/envs/iguane_env/share

RUN conda env create -f MURD/murd_env.yml && \
    conda clean --all --yes && \
    rm -rf /opt/conda/pkgs/* && \
    rm -rf /root/.cache/pip && \
    rm -rf /opt/conda/envs/murd_env/conda-meta \
    /opt/conda/envs/murd_env/include \
    /opt/conda/envs/murd_env/share


# 6) Installer click dans l'env de base pour la CLI
RUN conda install -n base click && conda clean -afy

# 7) Copier la CLI
WORKDIR /opt/toolkit
COPY cli/neuroharmo.py /opt/toolkit/neuroharmo.py
RUN chmod +x /opt/toolkit/neuroharmo.py

# 8) Entrypoint sur votre CLI
ENTRYPOINT ["/opt/toolkit/neuroharmo.py"]
CMD ["--help"]

# IGUANE
# docker run --rm \
#   -v /NAS/coolio/protocoles/galan/data/bids:/input \
#   -v /NAS/coolio/Barnabe/CODES/test_docker:/output \
#   neuroharmo_toolkit \
#     iguane \
#     /input/sub-00005/ses-20151211/anat/sub-00005_ses-20151211_acq-201T1_3D_TFE3MIN3D_T1w.nii.gz \
#     --apply_preproc_steps True \
#     --output_dir /output \
#     --save_preprocess True

# HACA3
# docker run --rm --privileged \
#   -v /NAS/coolio/protocoles/galan/data/bids:/input \
#   -v /NAS/coolio/Barnabe/CODES/test_docker:/output \
#   neuroharmo_toolkit \
#     haca3 \
#     /input/sub-00005/ses-20151211/anat/sub-00005_ses-20151211_acq-201T1_3D_TFE3MIN3D_T1w.nii.gz \
#     --apply_preproc_steps True \
#     --theta 10.0 --theta 20.0 \
#     --output_dir /output \
#     --save_preprocess True

# MURD
# docker run --rm \
#   -v /NAS/coolio/protocoles/galan/data/bids:/input \
#   -v /NAS/coolio/Barnabe/CODES/test_docker:/output \
#   neuroharmo_toolkit \
#     murd \
#     /input/sub-00005/ses-20151211/anat/sub-00005_ses-20151211_acq-201T1_3D_TFE3MIN3D_T1w.nii.gz \
#     --n_axial_slices 50 \
#     --output_dir /output \
#     --save_preprocess True

# STGAN
# docker run --rm \
#   -v /NAS/coolio/protocoles/galan/data/bids:/input \
#   -v /NAS/coolio/Barnabe/CODES/test_docker:/output \
#   neuroharmo_toolkit \
#     stgan \
#     /input/sub-00005/ses-20151211/anat/sub-00005_ses-20151211_acq-201T1_3D_TFE3MIN3D_T1w.nii.gz \
#     --apply_preproc_steps True \
#     --output_dir /output \
#     --save_preprocess True

# DISARMpp
# docker run --rm \
#   -v /NAS/coolio/protocoles/galan/data/bids:/input \
#   -v /NAS/coolio/Barnabe/CODES/test_docker:/output \
#   neuroharmo_toolkit \
#     disarmpp \
#     /input/sub-00005/ses-20151211/anat/sub-00005_ses-20151211_acq-201T1_3D_TFE3MIN3D_T1w.nii.gz \
#     --apply_preproc_steps True \
#     --output_dir /output \
#     --save_preprocess True
