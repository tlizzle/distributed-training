# Specifies base image and tag
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-5
WORKDIR /root

# RUN apt-get update -qq && \
#     apt-get install -yqq apt-utils && \
#     apt-get install -yqq gcc &&\
#     apt-get install -yqq graphviz &&\
#     apt-get install -yqq make &&\
#     apt-get install -yqq --no-install-recommends vim &&\
    
#     #Clean-up
#     rm -rf /var/lib/apt/lists/* && \
#     apt-get clean

# RUN pip install --upgrade pip
# RUN pip install pipenv
# RUN pipenv install --system --deploy --ignore-pipfile


# Copies the trainer code to the docker image.
COPY trainer/ /root/trainer/
COPY resources/ /root/resources

 
# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.main"]