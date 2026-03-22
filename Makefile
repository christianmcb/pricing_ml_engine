PYTHON=python

MODEL_ROOT=models
REGISTRY=$(MODEL_ROOT)/registry
CURRENT=$(MODEL_ROOT)/current

.PHONY: train predict api test docker-build docker-run list-models evaluate promote promote-latest clean-current simulate live-infer monitor retrain

train:
	$(PYTHON) -m scripts.train

predict:
	$(PYTHON) -m scripts.predict

api:
	$(PYTHON) -m scripts.serve_api

test:
	$(PYTHON) -m pytest tests/

docker-build:
	docker build -t pricing-ml-engine .

docker-run:
	docker run -p 8000:8000 pricing-ml-engine


list-models:
	@echo "Available trained model runs:"
	@ls -1 $(REGISTRY)


evaluate:
	@if [ -z "$(RUN_ID)" ]; then \
		echo "Usage: make evaluate RUN_ID=<model_run>"; \
		exit 1; \
	fi
	@if [ ! -d "$(REGISTRY)/$(RUN_ID)" ]; then \
		echo "Run $(RUN_ID) not found."; \
		exit 1; \
	fi
	@echo "Evaluating model $(RUN_ID)..."
	$(PYTHON) -m src.evaluate_model --run_id $(RUN_ID)


promote: evaluate
	@mkdir -p $(CURRENT)
	cp $(REGISTRY)/$(RUN_ID)/model.joblib $(CURRENT)/model.joblib
	cp $(REGISTRY)/$(RUN_ID)/model_metadata.json $(CURRENT)/model_metadata.json
	cp $(REGISTRY)/$(RUN_ID)/model_comparison.csv $(CURRENT)/model_comparison.csv
	cp $(REGISTRY)/$(RUN_ID)/best_params.json $(CURRENT)/best_params.json
	@if [ -f "$(REGISTRY)/$(RUN_ID)/feature_importance.csv" ]; then \
		cp $(REGISTRY)/$(RUN_ID)/feature_importance.csv $(CURRENT)/feature_importance.csv; \
	fi
	@echo "Model $(RUN_ID) promoted → $(CURRENT)"


promote-latest:
	@LATEST=$$(ls -1 $(REGISTRY) | sort | tail -n 1); \
	if [ -z "$$LATEST" ]; then \
		echo "No models found."; \
		exit 1; \
	fi; \
	$(MAKE) promote RUN_ID=$$LATEST


clean-current:
	rm -rf $(CURRENT)


simulate:
	$(PYTHON) -m scripts.simulate_live_data
	@echo '{"processed_batches": []}' > outputs/live_inference_state.json
	@echo "Live inference state reset."


live-infer:
	$(PYTHON) -m scripts.run_live_inference


monitor:
	$(PYTHON) -m scripts.monitor


retrain:
	$(PYTHON) -m scripts.retrain_if_needed