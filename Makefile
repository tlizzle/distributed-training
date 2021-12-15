PROJECT_ID=tensorflow-project
IMAGE_REPO_NAME=distributed-training
IMAGE_TAG=0.0.1
IMAGE_NAME=asia.gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}
# IMAGE_NAME=asia.gcr.io/mi3-cloud/traffic-profiling:0.0.1


all: | delete build push

build ::
	docker build -t ${IMAGE_NAME} .

push ::
	docker push ${IMAGE_NAME}

delete ::
	docker rmi ${IMAGE_NAME}
