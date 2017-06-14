FROM andrewosh/binder-base

# for use with mybinder.org

MAINTAINER Daniel Tamayo <tamayo.daniel@gmail.com>

USER main
#COPY . $HOME/

#RUN pip install rebound sympy matplotlib numpy scipy
RUN /home/main/anaconda/envs/python3/bin/conda install -c conda-forge #ipywidgets matplotlib numpy scipy
RUN /home/main/anaconda/envs/python3/bin/pip install rebound
