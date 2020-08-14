.ONESHELL: ;
.PHONY: build build_deployment_dependencies init test deploy clean

ZIP_FILE = handler.zip
function_name = yourLambdaFunctionNameHere

build: clean build_deployment_dependencies
	cd deployment; zip -r9 ../$(ZIP_FILE) . -x "tests/*" "schema/*" "pkgs/*"
	cd deployment/pkgs; zip -ur9 ../../$(ZIP_FILE) .
	rm -rf deployment/pkgs

	aws lambda update-function-code \
		--profile default \
		--function-name $(function_name)
		--zip-file "fileb//handler.zip"

build_deployment_dependencies:
	docker build -t deployment_dependencies .
	docker create --name dummy_container -t deployment_dependencies
	docker cp dummy_container:/build/pkgs deployment/
	docker rm -f dummy_container

init:
	pip install -r requirements.txt

test:
	py.test -s

clean:
	find . -name .pytest_cache -type d -print0 | xargs -0 rm -r --
	find . -name __pycache__ -type d -print0 | xargs -0 rm -r --
	rm -rf deployment/pkgs

