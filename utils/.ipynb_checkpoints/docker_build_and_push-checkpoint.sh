# ビルド
# <your_account_id>はアカウントid
# アカウントid：692139951728
docker build -t 692139951728.dkr.ecr.ap-northeast-1.amazonaws.com/fuyu:pytroch -f Dockerfile .
# ECRにプッシュ
docker push 692139951728.dkr.ecr.ap-northeast-1.amazonaws.com/fuyu:pytroch