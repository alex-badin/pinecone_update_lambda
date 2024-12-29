FROM public.ecr.aws/lambda/python:3.11

# Copy requirements and code
COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY src ${LAMBDA_TASK_ROOT}/src/

# Install dependencies
RUN pip install \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --python-version 3.11 \
    --only-binary=:all: \
    --no-deps \
    numpy==1.24.3 \
    tokenizers==0.15.2 \
    --target ${LAMBDA_TASK_ROOT}

# Install remaining dependencies
RUN pip install \
    -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt \
    --target ${LAMBDA_TASK_ROOT}

# Create deployment package
RUN cd ${LAMBDA_TASK_ROOT} && zip -r /deployment_package.zip .