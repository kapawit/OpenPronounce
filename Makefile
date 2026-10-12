.PHONY: help build build-gpu save-image save-image-gpu run run-gpu test clean

HF_TOKEN := $(shell grep HF_TOKEN .env 2>/dev/null | cut -d= -f2)

help:
	@echo "OpenPronounce Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make build          Build CPU image"
	@echo "  make build-gpu     Build GPU image"
	@echo "  make save-image    Save CPU image to /mnt/data/docker-images/"
	@echo "  make save-image-gpu Save GPU image to /mnt/data/docker-images/"
	@echo "  make run           Run CPU container"
	@echo "  make run-gpu       Run GPU container"
	@echo "  make test          Run tests"
	@echo "  make clean         Remove containers"
	@echo ""
	@echo "Variables:"
	@echo "  HF_TOKEN    =$(HF_TOKEN)"

build:
	podman build --build-arg HF_TOKEN=$(HF_TOKEN) -t openpronounce .

build-gpu:
	podman build --build-arg HF_TOKEN=$(HF_TOKEN) -f Dockerfile.gpu -t openpronounce:gpu .

save-image:
	mkdir -p /mnt/data/docker-images
	podman save openpronounce | gzip > /mnt/data/docker-images/openpronounce.tar.gz

save-image-gpu:
	mkdir -p /mnt/data/docker-images
	podman save openpronounce:gpu | gzip > /mnt/data/docker-images/openpronounce-gpu.tar.gz

run:
	podman run -p 8000:8000 --name openpronounce openpronounce

run-gpu:
	podman run --gpus all -p 8000:8000 --name openpronounce-gpu openpronounce:gpu

test:
	.venv/bin/python -m pytest tests/ -v

clean:
	podman stop openpronounce openpronounce-gpu 2>/dev/null || true
	podman rm openpronounce openpronounce-gpu 2>/dev/null || true
