FROM andrewosh/binder-base

# for use with mybinder.org

MAINTAINER Sam Hadden <shadden1107@gmail.com>

USER root
COPY . $HOME/

RUN pip install rebound celmech
RUN $HOME/anaconda2/envs/python3/bin/pip install rebound celmech
