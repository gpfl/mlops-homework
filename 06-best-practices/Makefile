setup:
	pipenv install

unit-tests:
	pipenv run pytest tests/

integration-tests:
	pipenv run python integration-test/integration_test.py

s3-up:
	docker-compose up localstack

create-bucket:
	aws --endpoint-url=${S3_ENDPOINT_URL} s3 mb s3://nyc-duration

s3-ls:
	aws --endpoint-url=${S3_ENDPOINT_URL} s3 ls nyc-duration/