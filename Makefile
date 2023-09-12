SHELL:=/bin/bash
VENV=.venv
PYTHON_VERSION=3.10
PYTHON=$(VENV)/bin/python3
PIP=$(VENV)/bin/pip
ACT="./bin/act"
MAKEFLAGS += --no-print-directory

IS_DEBUG=0
IS_MISS_JBSUB=$(shell which jbsub &>/dev/null && echo 0 || echo 1)

LOG_DIRNAME=./logs
DATASETS_DIRNAME=./datasets
AUTOML_DIRNAME=automl_search

GPUS=0
CORES=6
MEMORY=32
EPOCHS=1000
PATIENCE=20
TIME_LIMIT=40000
AUTOML_TRIALS=1000
SUBSETS_TRIALS=1000
MODEL_PRUNING_TRIALS=10000
MODELS_PRUNED_FOR_SUBSETS_MAX=100
PERFORMANCE_DEGRADATION=0.10
EVAL_METRIC="accuracy"
PLOT_FORMAT="png" # or "svg"
MODEL_NAME="best_model"

COMPOSITION= "balanced"#"balanced" "unbalanced"
DATASETS= "CICIDS2017" "ICS-D1-BIN" "EDGE2022"#"EDGE2022" "CICIOT2023" "CICIDS2017" "CICIDS2019" "ICS-D1-BIN"
FEATURE_RANKING_ALGORITHMS= "custom"#"custom" "pca" "autogluon" "random_forest"
FEATURE_RANKING_STRATEGIES= "sequential_backward_elimination" #"sequential_backward_elimination" "sequential_forward_selection"
SUBSETS_SIZE_AMOUNTS=                   0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95
BEST_FEATURES_FOR_SUBSET_AMOUNTS=  0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00
MODEL_PRUNING_AMOUNTS=             0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00
MODEL_PRUNING_ALGORITHMS= "locally_connections_l1" "locally_neurons_l1" "locally_connections_random"#"locally_connections_l1" "locally_connections_l2" "locally_connections_random"\
								"locally_neurons_l1" "locally_neurons_l2" "locally_neurons_random" \
								"locally_neurons_activation_l1" "locally_neurons_activation_l2"\
								"locally_neurons_activation_l1_for_subset" "locally_neurons_activation_l2_for_subset"\
								"globally_connections_l1" "globally_connections_l2" "globally_connections_random"\
								"globally_neurons_l1" "globally_neurons_l2" "globally_neurons_random"\
								"globally_neurons_activation_l1" "globally_neurons_activation_l2"\
								"globally_neurons_activation_l1_for_subset" "globally_neurons_activation_l2_for_subset"
SUBSET_LEFT_TAKEN_POLICIES= "random"# "weighted"
PLOT_METRICS= "accuracy"# "f1score" ...

####################################################
################## VENV SPECIFIC ###################
####################################################
	
create: requirements.txt
	@if ! [ -d $(VENV) ]; then\
        python${PYTHON_VERSION} -m venv $(VENV);\
		$(PIP) install --upgrade pip;\
		$(PIP) install -r requirements.txt;\
	fi

create-dev: create
	$(PIP) install pycodestyle==2.10.0
	$(PIP) install pylint==2.17.4
	$(PIP) install autopep8==2.0.2
	curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

update:
	$(PIP) install -r requirements.txt

clean:
	rm -rf __pycache__
	rm -rf $(VENV)

.PHONY: run clean


####################################################
################## TEST SPECIFIC ###################
####################################################

# https://github.com/nektos/act
test-cicd-lint-local:
	${ACT} -j markdown-lint -W .github/workflows/doc.yml

test-cicd-code-local:
	${ACT} -j python-lint -W .github/workflows/code.yml

test-code:
	$(PYTHON) -m pycodestyle intellect
	$(PYTHON) -m pylint intellect

test-lint: test-cicd-lint

#######################################################
################## HELPER TO FORMAT ###################
#######################################################

create-dir:
    ifndef DIR_TO_CREATE
	$(error DIR_TO_CREATE is undefined)
    else
	@mkdir -p ${DIR_TO_CREATE}
    endif

get-cmd-formatted:
    ifndef CMD_NAME
	$(error CMD_NAME is undefined)
    endif
    ifndef DS_NAME
	$(error DS_NAME is undefined)
    endif
    ifndef COMPOSITION_TYPE
	$(error COMPOSITION_TYPE is undefined)
    endif
    ifeq (${IS_MISS_JBSUB}, 1)
    ifeq (${IS_DEBUG},1)
	@DATETIME=$$(date '+%Y-%m-%d-%H:%M:%S') && echo "echo ''@'2>&1 | tee ${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE}/${CMD_NAME}/$${DATETIME}.log'"
    else
	@${MAKE} DIR_TO_CREATE=${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE}/${CMD_NAME} create-dir
	@DATETIME=$$(date '+%Y-%m-%d-%H:%M:%S') && echo ": && @2>&1 | tee ${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE}/${CMD_NAME}/$${DATETIME}.log"
    endif
    else
    ifeq (${IS_DEBUG},1)
	@DATETIME=$$(date '+%Y-%m-%d-%H:%M:%S') && echo echo "jbsub -cores 1x${CORES}+${GPUS} -mem ${MEMORY}G -out ${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE}/${CMD_NAME}/$${DATETIME}.jbinfo -err ${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE}/${CMD_NAME}/$${DATETIME}.log @ && :"
    else
	@${MAKE} DIR_TO_CREATE=${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE}/${CMD_NAME} create-dir
	@DATETIME=$$(date '+%Y-%m-%d-%H:%M:%S') && echo "jbsub -cores 1x${CORES}+${GPUS} -mem ${MEMORY}G -out ${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE}/${CMD_NAME}/$${DATETIME}.jbinfo -err ${LOG_DIRNAME}/${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE}/${CMD_NAME}/$${DATETIME}.log @ && :"
    endif
    endif

#######################################################
## RUN SINGLE EITHER LOCAL OR FROM JBSUB-ENABLED ENV ##
#######################################################

run-dataset-creator:
    ifndef DS_NAME
	$(error DS_NAME is undefined)
    endif
    ifndef COMPOSITION_TYPE
	$(error COMPOSITION_TYPE is undefined)
    endif
	@TMP=$$(${MAKE} DS_NAME=${DS_NAME} COMPOSITION_TYPE=${COMPOSITION_TYPE} CMD_NAME=${MAKECMDGOALS} get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/dataset_creator.py ${DATASETS_DIRNAME}/${DS_NAME}/original ${COMPOSITION_TYPE} $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-automl-search:
    ifndef DS_NAME
	$(error DS_NAME is undefined)
    endif
    ifndef COMPOSITION_TYPE
	$(error COMPOSITION_TYPE is undefined)
    endif
	@TMP=$$(${MAKE} DS_NAME=${DS_NAME} COMPOSITION_TYPE=${COMPOSITION_TYPE} CMD_NAME=${MAKECMDGOALS}-${TIME_LIMIT}t-${EVAL_METRIC}m-${AUTOML_TRIALS}a-${PATIENCE}p-${EPOCHS}e get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/automl_model_search.py search ${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE} -c ${CORES} -g ${GPUS} -a ${AUTOML_TRIALS} -e ${EPOCHS} -p ${PATIENCE} -t ${TIME_LIMIT} -m ${EVAL_METRIC} $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-automl-dump:
    ifndef DS_NAME
	$(error DS_NAME is undefined)
    endif
    ifndef COMPOSITION_TYPE
	$(error COMPOSITION_TYPE is undefined)
    endif
	@TMP=$$(${MAKE} DS_NAME=${DS_NAME} COMPOSITION_TYPE=${COMPOSITION_TYPE} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME} get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/automl_model_search.py dump ${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE}/${AUTOML_DIRNAME} -m ${MODEL_NAME} $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-feature-ranking:
    ifndef DS_NAME
	$(error DS_NAME is undefined)
    endif
    ifndef COMPOSITION_TYPE
	$(error COMPOSITION_TYPE is undefined)
    endif
    ifndef ALGORITHM
	$(error ALGORITHM is undefined)
    endif
    ifndef STRATEGY
	$(error STRATEGY is undefined)
    endif
	@TMP=$$(${MAKE} DS_NAME=${DS_NAME} COMPOSITION_TYPE=${COMPOSITION_TYPE} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${ALGORITHM}-${STRATEGY}-${TIME_LIMIT}t-${EVAL_METRIC} get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/feature_ranking.py rank ${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE}/${AUTOML_DIRNAME}/models/${MODEL_NAME} ${ALGORITHM} ${STRATEGY} -c ${CORES} -t ${TIME_LIMIT} -m ${EVAL_METRIC} $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-feature-subset-stochastic-search:
    ifndef DS_NAME
	$(error DS_NAME is undefined)
    endif
    ifndef COMPOSITION_TYPE
	$(error COMPOSITION_TYPE is undefined)
    endif
    ifndef ALGORITHM
	$(error ALGORITHM is undefined)
    endif
    ifndef STRATEGY
	$(error STRATEGY is undefined)
    endif
    ifndef TYPE_SEARCH
	$(error TYPE_SEARCH is undefined)
    endif
    ifndef SIZE
	$(error SIZE is undefined)
    endif
	@TMP=$$(${MAKE} DS_NAME=${DS_NAME} COMPOSITION_TYPE=${COMPOSITION_TYPE} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${ALGORITHM}-${STRATEGY}-${TYPE_SEARCH}-${SIZE}s-${EVAL_METRIC} get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/feature_subset_stochastic_search.py search ${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE}/${AUTOML_DIRNAME}/models/${MODEL_NAME}/feature_ranking_${STRATEGY}_${ALGORITHM} ${SIZE} ${TYPE_SEARCH} -a ${SUBSETS_TRIALS} -b ${BEST_FEATURES_FOR_SUBSET_AMOUNTS} -c ${CORES} -m ${EVAL_METRIC}$${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-model-pruner-search:
    ifndef DS_NAME
	$(error DS_NAME is undefined)
    endif
    ifndef COMPOSITION_TYPE
	$(error COMPOSITION_TYPE is undefined)
    endif
    ifndef PRUNE_ALGORITHM
	$(error PRUNE_ALGORITHM is undefined)
    endif
    ifndef SUBSET_TARGET
	SUBSET_TARGET="";
    endif
	@TMP=$$(${MAKE} DS_NAME=${DS_NAME} COMPOSITION_TYPE=${COMPOSITION_TYPE} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${PRUNE_ALGORITHM}-${PERFORMANCE_DEGRADATION}d-${EVAL_METRIC}-search get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/model_pruner_search.py search ${DATASETS_DIRNAME}/$${DS_NAME}/${COMPOSITION_TYPE}/${AUTOML_DIRNAME}/models/${MODEL_NAME} ${PRUNE_ALGORITHM} -a ${MODEL_PRUNING_TRIALS} -p ${MODEL_PRUNING_AMOUNTS} -c ${CORES} -s $${SUBSET_TARGET} -d ${PERFORMANCE_DEGRADATION} -m ${EVAL_METRIC} -t ${MODELS_PRUNED_FOR_SUBSETS_MAX} $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-model-pruner-test:
    ifndef DS_NAME
	$(error DS_NAME is undefined)
    endif
    ifndef COMPOSITION_TYPE
	$(error COMPOSITION_TYPE is undefined)
    endif
    ifndef ALGORITHM
	$(error ALGORITHM is undefined)
    endif
    ifndef STRATEGY
	$(error STRATEGY is undefined)
    endif
    ifndef TYPE_SEARCH
	$(error TYPE_SEARCH is undefined)
    endif
    ifndef SIZE
	$(error SIZE is undefined)
    endif
    ifndef PRUNE_ALGORITHM
	$(error PRUNE_ALGORITHM is undefined)
    endif
	@TMP=$$(${MAKE} DS_NAME=${DS_NAME} COMPOSITION_TYPE=${COMPOSITION_TYPE} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${ALGORITHM}-${STRATEGY}-${TYPE_SEARCH}-${SIZE}s-${PRUNE_ALGORITHM}-${PERFORMANCE_DEGRADATION}d-${EVAL_METRIC}-test get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/model_pruner_search.py test ${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE}/${AUTOML_DIRNAME}/models/${MODEL_NAME}/feature_ranking_${STRATEGY}_${ALGORITHM}/feature_subset_stochastic_search_${TYPE_SEARCH}/feature_subsets_${SIZE}s ${PRUNE_ALGORITHM} -c ${CORES} -d ${PERFORMANCE_DEGRADATION} -m ${EVAL_METRIC} -t ${MODELS_PRUNED_FOR_SUBSETS_MAX} $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-plot-feature-ranking:
    ifndef DS_NAME
	$(error DS_NAME is undefined)
    endif
    ifndef COMPOSITION_TYPE
	$(error COMPOSITION_TYPE is undefined)
    endif
    ifndef ALGORITHM
	$(error ALGORITHM is undefined)
    endif
    ifndef STRATEGY
	$(error STRATEGY is undefined)
    endif
    ifndef METRIC
	$(error METRIC is undefined)
    endif
	@TMP=$$(${MAKE} DS_NAME=${DS_NAME} COMPOSITION_TYPE=${COMPOSITION_TYPE} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${METRIC}-plot get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/feature_ranking.py plot ${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE}/${AUTOML_DIRNAME}/models/${MODEL_NAME}/feature_ranking_${STRATEGY}_${ALGORITHM} -m ${METRIC} -f ${PLOT_FORMAT} $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-plot-feature-subset-stochastic-search:
    ifndef DS_NAME
	$(error DS_NAME is undefined)
    endif
    ifndef COMPOSITION_TYPE
	$(error COMPOSITION_TYPE is undefined)
    endif
    ifndef ALGORITHM
	$(error ALGORITHM is undefined)
    endif
    ifndef STRATEGY
	$(error STRATEGY is undefined)
    endif
    ifndef TAKEN_POLICY
	$(error TAKEN_POLICY is undefined)
    endif
    ifndef METRIC
	$(error METRIC is undefined)
    endif
	@TMP=$$(${MAKE} DS_NAME=${DS_NAME} COMPOSITION_TYPE=${COMPOSITION_TYPE} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${METRIC}-plot get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/feature_subset_stochastic_search.py plot ${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE}/${AUTOML_DIRNAME}/models/${MODEL_NAME}/feature_ranking_${STRATEGY}_${ALGORITHM}/feature_subset_stochastic_search_${TAKEN_POLICY} -m ${METRIC} -f ${PLOT_FORMAT} $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-plot-model-pruner-search:
    ifndef DS_NAME
	$(error DS_NAME is undefined)
    endif
    ifndef COMPOSITION_TYPE
	$(error COMPOSITION_TYPE is undefined)
    endif
    ifndef METRIC
	$(error METRIC is undefined)
    endif
	@TMP=$$(${MAKE} DS_NAME=${DS_NAME} COMPOSITION_TYPE=${COMPOSITION_TYPE} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${METRIC}-plot-search get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/model_pruner_search.py plot_search ${DATASETS_DIRNAME}/${DS_NAME}/${COMPOSITION_TYPE}/${AUTOML_DIRNAME}/models/${MODEL_NAME}/prune_search/ -s ${SUBSETS_DIR} -m ${METRIC} -f ${PLOT_FORMAT} $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

run-plot-model-pruner-test:
    ifndef DIR_OF_INTEREST
	$(error DIR_OF_INTEREST is undefined)
    endif
    ifndef METRIC
	$(error METRIC is undefined)
    endif
    ifndef PRUNE_ALGORITHM
	$(error PRUNE_ALGORITHM is undefined)
    endif
	@TMP=$$(${MAKE} DS_NAME=${DS_NAME} COMPOSITION_TYPE=${COMPOSITION_TYPE} CMD_NAME=${MAKECMDGOALS}-${MODEL_NAME}-${PRUNE_ALGORITHM}-${METRIC}-plot-test get-cmd-formatted);\
	if [ $$? -ne 0 ]; then\
		echo "Error formatting the command (Read above)";\
		exit 1;\
	fi;\
	IFS=@ read -r PREFIX SUFFIX <<< "$${TMP}";\
	CMD="$${PREFIX} ${PYTHON} scripts/model_pruner_search.py plot_test ${DIR_OF_INTEREST} ${PRUNE_ALGORITHM} -m ${METRIC} -f {PLOT_FORMAT} $${SUFFIX}";\
	echo -e "@@@ Going to Run -> $${CMD}\n\n";\
	eval $${CMD}

#######################################################
##### RUN EITHER LOCAL OR FROM JBSUB-ENABLED ENV ######
#######################################################

run-all-dataset-creator:
	for f in ${DATASETS}; do\
		for ct in ${COMPOSITION}; do\
			${MAKE} DS_NAME=$${f} COMPOSITION_TYPE=$${ct} run-dataset-creator;\
		done;\
	done

run-all-automl-search:
	for f in ${DATASETS}; do\
		for ct in ${COMPOSITION}; do\
			${MAKE} DS_NAME=$${f} COMPOSITION_TYPE=$${ct} run-automl-search;\
		done;\
	done

run-all-automl-dump:
	for f in ${DATASETS}; do\
		for ct in ${COMPOSITION}; do\
			${MAKE} DS_NAME=$${f} COMPOSITION_TYPE=$${ct} run-automl-dump;\
		done;\
	done

run-all-feature-ranking:
	for f in ${DATASETS}; do\
		for ct in ${COMPOSITION}; do\
			for alg in ${FEATURE_RANKING_ALGORITHMS}; do\
				for st in ${FEATURE_RANKING_STRATEGIES}; do\
					${MAKE} DS_NAME=$${f} COMPOSITION_TYPE=$${ct} ALGORITHM=$${alg} STRATEGY=$${st} run-feature-ranking;\
				done;\
			done;\
		done;\
	done

run-all-feature-subset-stochastic-search:
	for f in ${DATASETS}; do\
		for ct in ${COMPOSITION}; do\
			for alg in ${FEATURE_RANKING_ALGORITHMS}; do\
				for st in ${FEATURE_RANKING_STRATEGIES}; do\
					for t in ${SUBSET_LEFT_TAKEN_POLICIES}; do\
						for s in ${SUBSETS_SIZE_AMOUNTS}; do\
							${MAKE} DS_NAME=$${f} COMPOSITION_TYPE=$${ct} ALGORITHM=$${alg} STRATEGY=$${st} TYPE_SEARCH=$${t} SIZE=$${s} run-feature-subset-stochastic-search;\
						done;\
					done;\
				done;\
			done;\
		done;\
	done

run-all-model-pruner-search:
	for f in ${DATASETS}; do\
		for ct in ${COMPOSITION}; do\
			for malg in ${MODEL_PRUNING_ALGORITHMS}; do\
				if [[ $${malg} == *"_for_subset"* ]]; then\
					for alg in ${FEATURE_RANKING_ALGORITHMS}; do\
						for st in ${FEATURE_RANKING_STRATEGIES}; do\
							for t in ${SUBSET_LEFT_TAKEN_POLICIES}; do\
								for s in ${SUBSETS_SIZE_AMOUNTS}; do\
									${MAKE} DS_NAME=$${f} COMPOSITION_TYPE=$${ct} PRUNE_ALGORITHM=$${malg} SUBSET_TARGET=${DATASETS_DIRNAME}/$${f}/$${ct}/${AUTOML_DIRNAME}/models/${MODEL_NAME}/feature_ranking_$${st}_$${alg}/feature_subset_stochastic_search_$${t}/feature_subsets_$${s}s run-model-pruner-search;\
								done;\
							done;\
						done;\
					done;\
				else\
					${MAKE} DS_NAME=$${f} COMPOSITION_TYPE=$${ct} PRUNE_ALGORITHM=$${malg} SUBSET_TARGET="" run-model-pruner-search;\
				fi;\
			done;\
		done;\
	done

run-all-model-pruner-test:
	for f in ${DATASETS}; do\
		for ct in ${COMPOSITION}; do\
			for st in ${FEATURE_RANKING_STRATEGIES}; do\
				for ralg in ${FEATURE_RANKING_ALGORITHMS}; do\
					for t in ${SUBSET_LEFT_TAKEN_POLICIES}; do\
						for alg in ${MODEL_PRUNING_ALGORITHMS}; do\
							for s in ${SUBSETS_SIZE_AMOUNTS}; do\
								${MAKE} DS_NAME=$${f} COMPOSITION_TYPE=$${ct} SIZE=$${s} STRATEGY=$${st} ALGORITHM=$${ralg} PRUNE_ALGORITHM=$${alg} TYPE_SEARCH=$${t} run-model-pruner-test;\
							done;\
						done;\
					done;\
				done;\
			done;\
		done;\
	done


run-all-print:
	for f in ${DATASETS}; do\
		for ct in ${COMPOSITION}; do\
			for met in ${PLOT_METRICS}; do\
				for st in ${FEATURE_RANKING_STRATEGIES}; do\
					for a in ${FEATURE_RANKING_ALGORITHMS}; do\
						${MAKE} DS_NAME=$${f} COMPOSITION_TYPE=$${ct} METRIC=$${met} STRATEGY=$${st} ALGORITHM=$${a} run-plot-feature-ranking;\
						for t in ${SUBSET_LEFT_TAKEN_POLICIES}; do\
							${MAKE} DS_NAME=$${f} COMPOSITION_TYPE=$${ct} METRIC=$${met} STRATEGY=$${st} ALGORITHM=$${a} TAKEN_POLICY=$${t} run-plot-feature-subset-stochastic-search;\
							for alg in ${MODEL_PRUNING_ALGORITHMS}; do\
								echo "";\
							done;\
						done;\
					done;\
				done;\
				${MAKE} DS_NAME=$${f} COMPOSITION_TYPE=$${ct} METRIC=$${met} SUBSETS_DIR=$${SUBSETS} run-plot-model-pruner-search;\
			done;\
		done;\
	done

# ${MAKE} DS_NAME=$${f} COMPOSITION_TYPE=$${ct} METRIC=$${met} PRUNE_ALGORITHM=$${alg} run-plot-model-pruner-test;

check-jobs:
    ifeq (${IS_MISS_JBSUB}, 1)
	@echo "Currently" $$(( `ps aux | grep ${USER} | grep -e "automl_model_search" -e "dataset_creator" -e "feature_ranking" -e "feature_subset_stochastic_search" -e "model_pruner_search" | wc -l` - 3 )) "jobs"
    else
	@echo "Currently" $$(( `jbinfo -state run | grep ${USER} | wc -l` + `jbinfo -state pend | grep ${USER} | wc -l` )) "jobs"
    endif


stop-jobs:
    ifeq (${IS_MISS_JBSUB}, 1)
	@pkill -f automl_model_search.py || true;\
	pkill -f dataset_creator.py || true;\
	pkill -f feature_ranking.py || true;\
	pkill -f feature_subset_stochastic_search.py || true;\
	pkill -f model_pruner_search.py || true
    else
	@for jid in `jbinfo -state pend | awk 'FNR > 2 { print $$1 }'`; do\
		jbadmin -kill $${jid};\
	done;\
	for jid in `jbinfo -state run | awk 'FNR > 2 { print $$1 }'`; do\
		jbadmin -kill $${jid};\
	done
    endif
