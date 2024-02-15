MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_NAME=modelo.joblib
ALIASES=modelo

train:
	python src/train/hyperparameter.py

format:
	ruff check --select I --fix . && ruff format .

deploy_aws:
	cd src/deploy/aws; python deployment.py

inference_aws:
	cd src/deploy/aws; python3 inference.py

inference_azure:
	cd src/deploy/azure; python3 inference.py

monitoring:
	cd src/monitoring; python3 monitor.py

generate-dockerfile:
	export MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	&& cd .. \
	&& mlflow models generate-dockerfile --model-uri "models:/$(MODEL_NAME)@$(ALIASES)"

doc:
	mkdocs serve

doc-deploy:
	mkdocs gh-deploy