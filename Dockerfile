FROM python:3.7

COPY ./dependencies /app/dependencies
COPY ./emlopt /app/emlopt
COPY ./tests /app/tests
COPY ./notebooks /app/notebooks
COPY ./problems /app/problems

WORKDIR /app

RUN apt-get update -y && apt-get install zip -y
RUN apt-get install wget git -y
RUN pip install --upgrade pip 

# pip deps
RUN pip install -r dependencies/requirements.txt \
    RISE jupyter_contrib_nbextensions tables tensorflow_probability
RUN jupyter contrib nbextension install --system

# eml
RUN git clone https://github.com/DanieleVeri/emllib.git dependencies/emllib

# cplex
RUN mkdir dependencies/cplex
RUN wget https://api.wandb.ai/artifactsV2/gcp-us/veri/QXJ0aWZhY3Q6MjU1ODAzMjI=/69b1b89a73a7d0931fbfdb355eb147c3 -O dependencies/cplex/cplex_studio1210.linux-x86-64.bin
RUN wget https://api.wandb.ai/artifactsV2/gcp-us/veri/QXJ0aWZhY3Q6MjU1ODAzMjI=/97133b747b0114a4e3dba77ab26d68d5 -O dependencies/cplex/response.properties
RUN pip install docplex
RUN sh dependencies/cplex/cplex_studio1210.linux-x86-64.bin -f response.properties
RUN python3 /opt/ibm/ILOG/CPLEX_Studio1210/python/setup.py install

WORKDIR /app

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", \
    "--ip=0.0.0.0", "--allow-root", "--notebook-dir=notebooks"]