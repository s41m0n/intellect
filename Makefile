SHELL:=/bin/bash
VENV=.venv
PYTHON_VERSION=3.10
PYTHON=$(VENV)/bin/python3
PIP=$(VENV)/bin/pip
ACT="./bin/act"

install: 
	@if ! [ -d $(VENV) ]; then\
        python${PYTHON_VERSION} -m venv $(VENV);\
		$(PIP) install --upgrade pip;\
	fi
	$(PIP) install .;\

install-dev: install
	$(PIP) install --editable ".[test]" 
	curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

install-automl: install
	$(PIP) install --no-deps --editable ".[automl]" 

update:
	$(PIP) install --upgrade .

test-cicd-lint-local:
	${ACT} -j markdown-lint -W .github/workflows/doc.yml

test-cicd-code-local:
	${ACT} -j python-lint -W .github/workflows/code.yml

test-code:
	$(PYTHON) -m pycodestyle src
	$(PYTHON) -m pylint src

test-lint: test-cicd-lint-local

clean:
	rm -rf __pycache__
	rm -rf $(VENV)

.PHONY: run clean