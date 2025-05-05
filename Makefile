# Image tags
RUNTIME_TAG=evabyte:runtime
DEV_TAG=evabyte:dev

# Dockerfile location
DOCKERFILE=Dockerfile

.PHONY: all build-runtime build-dev run-runtime run-dev shell-dev

all: build-runtime build-dev

build-runtime:
	docker build --target runtime -t $(RUNTIME_TAG) -f $(DOCKERFILE) .

build-dev:
	docker build --target dev -t $(DEV_TAG) -f $(DOCKERFILE) .

run-runtime:
	docker run --rm -it \
		-v $(PWD)/src:/job/src \
		-v $(PWD)/data:/job/data \
		-v $(PWD)/work:/job/work \
		-v $(PWD)/output:/job/output \
		--gpus all \
		$(RUNTIME_TAG)

run-dev:
	docker run --rm -it \
		-v $(PWD)/src:/job/src \
		-v $(PWD)/data:/job/data \
		-v $(PWD)/work:/job/work \
		-v $(PWD)/output:/job/output \
		-v $(PWD):/workspace \
		--gpus all \
		--name devcontainer \
		$(DEV_TAG)

shell-dev:
	docker exec -it devcontainer /bin/bash

