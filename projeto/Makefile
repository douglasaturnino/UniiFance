train:
	python src/train/train.py

format:
	ruff check --select I --fix . && ruff format .

deploy_aws:
	cd src/deploy/aws; python deployment.py

inference_aws:
	cd src/deploy/aws; python3 inference.py