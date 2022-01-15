PROJECT_ID=tensorflow-project-335114
IMAGE_REPO_NAME=distributed-training
IMAGE_TAG=0.0.1
IMAGE_NAME=asia.gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}


all: | delete build

build ::
	docker build -t ${IMAGE_NAME} .

push ::
	docker push ${IMAGE_NAME}

delete ::
	docker rmi ${IMAGE_NAME}

ai-submit ::
	gcloud ai-platform jobs submit training distributed_test_12 \
	--project tensorflow-project-335114 \
	--region asia-east1 \
	--staging-bucket gs://tessorflow-store \
	--config config.yaml \
	--job-dir gs://tessorflow-store/model_dir -- \
	--epochs 5

test :
	docker run --rm  70392ac3681f --epochs 5 --job-dir gs://tessorflow-store/model_dir

# ai-submit-local ::
# 	gcloud ai-platform jobs submit training distributed_test_4 \
# 	--project tensorflow-project-335114 \
# 	--region asia-east1 \
# 	--staging-bucket gs://tessorflow-store \
# 	--config config.yaml \
# 	--job-dir /Users/tony/Desktop/DS/distributed_training -- \
# 	--epochs 5

# test :
# 	docker run --rm -v ~/.config/gcloud/:/root/.config/gcloud  3ec5584f45ba --epochs 5 --job-dir gs://tessorflow-store/model_dir

