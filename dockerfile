# Specifies base image and tag
FROM gcr.io/deeplearning-platform-release/tf-cpu as base
USER root

WORKDIR /root


# Copies the trainer code to the docker image.
COPY trainer/ /root/trainer/
COPY resources/ /root/resources
COPY Pipfile /root/Pipfile
COPY Pipfile.lock /root/Pipfile.lock
ADD /gcloud/ /root/.config/gcloud


RUN apt-get update -qq && \
    apt-get install -yqq apt-utils && \
    apt-get install -yqq gcc &&\
    apt-get install -yqq graphviz &&\
    apt-get install -yqq make &&\
    apt-get install -yqq --no-install-recommends vim &&\
    
    #Clean-up
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN pip install --upgrade pip
RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile


# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]


