all: quality test
check_dirs := hallucinations_kg scripts tests
# Check that source code meets quality standards

quality:
	pre-commit run --all-files
	mypy --install-types --non-interactive $(check_dirs)

fix:
	pre-commit run --all-files

test:
	pytest .

install:
	pip install -r requirements.txt
