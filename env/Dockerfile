FROM python:3.7

WORKDIR /app

RUN mkdir dependencies
COPY ./env/requirements.txt /app/dependencies/requirements.txt

RUN apt-get update -y
RUN apt-get install zip wget git -y

# pip deps
RUN pip install --upgrade pip
RUN pip install -r dependencies/requirements.txt \
    RISE jupyter_contrib_nbextensions tables tensorflow_probability
RUN jupyter contrib nbextension install --system
# debug
RUN pip install ptvsd

# cplex
RUN mkdir dependencies/cplex
RUN wget https://api.wandb.ai/artifactsV2/gcp-us/veri/QXJ0aWZhY3Q6MjU1ODAzMjI=/69b1b89a73a7d0931fbfdb355eb147c3 -O dependencies/cplex/cplex_studio1210.linux-x86-64.bin
RUN wget https://api.wandb.ai/artifactsV2/gcp-us/veri/QXJ0aWZhY3Q6MjU1ODAzMjI=/97133b747b0114a4e3dba77ab26d68d5 -O dependencies/cplex/response.properties
RUN pip install docplex
RUN sh dependencies/cplex/cplex_studio1210.linux-x86-64.bin -f response.properties
RUN python3 /opt/ibm/ILOG/CPLEX_Studio1210/python/setup.py install


RUN mkdir env
COPY ./env/entrypoint.sh /app/env/entrypoint.sh
COPY ./env/launch_debug.sh /app/env/launch_debug.sh
COPY ./env/run_all_tests.py /app/env/run_all_tests.py

ENTRYPOINT ["/bin/bash", "/app/env/entrypoint.sh"]
