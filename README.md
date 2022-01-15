# Description
To utilize ai platform service provided by Google on gcp to implement distributed training. In this project, it presents the way to use tensorflow structure to build and train a simple logistic regression in a distributed manner

## Data
Use the well-known dataset `Iris.csv` with the purpose of identifing the type of flower.

y = Species

X = SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm

## Pre-requisite
- GCP account
- billing account enabled
- activate container registry (gcp)
- google credential
    - run `gcloud auth login` to get credential and store it `~/.config/gcloud/`
And this directory need to be pass into docker container in order to access the gcs (saving model)
[reference](https://stackoverflow.com/questions/53306131/difference-between-gcloud-auth-application-default-login-and-gcloud-auth-logi)

## Usage
gcloud ai-platform jobs submit training {job_name} \
--project {project_id} \
--region asia-east1 \
--staging-bucket {bucket_path} \
--config {config_yaml_path} \
--job-dir {model_save_path} -- \
--epochs 5



