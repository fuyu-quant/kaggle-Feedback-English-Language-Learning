# SageMakerからECRへログインする
aws ecr get-login-password | docker login --username AWS --password-stdin 692139951728.dkr.ecr.ap-northeast-1.amazonaws.com