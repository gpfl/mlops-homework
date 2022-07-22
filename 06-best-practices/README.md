# Homework 6 - Best Practices

## Code Reproduction
To reproduce this code, you need to have `pipenv`, `docker` and `docker-compose` installed.

### Prepare pipenv environment
```bash
make setup
```

### Environment variables
```bash
export AWS_ACCESS_KEY_ID=foobar
export AWS_SECRET_ACCESS_KEY=foobar
export S3_ENDPOINT_URL="http://localhost:4566"
export INPUT_FILE_PATTERN="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
```

### Run Unit tests
```bash
make unit-tests
```

### Run Localstack (AWS S3) with docker-compose 
```bash
make s3-up
```

### Create localstack S3 bucket `nyc-duration`
```bash
make s3-bucket
```

### Run integration tests
```bash
make integration-tests
```

### List files on localstack S3 bucket `nyc-duration`
```bash
make s3-ls
```