# Eppstein アルゴリズム

## 準備
- まず，cloneをする
## 構成
- dataディレクトリに
```
data
- dataset001
  - out
  graph001.txt
  pos001.txt
```


## image操作
### imageの確認
- `docker image ls`
- `docker images`
### imageのビルド
- `docker-compose build`
- `docker-compose build --no-cache`
### imageの削除
- docker image rm "IMAGE_ID" （IMAGE_IDは`docker images`で確認可能）

## コンテナ操作
### コンテナ起動
- `docker-compose up -d`
### コンテナ停止
- `docker compose down`
### コンテナ確認
- `docker compose ps`
- `docker container ls`
### コンテナ削除
- `docker rm "CONTAINER_ID"`

## execコマンド
コンテナ起動後に利用
- `docker exec -it eppstein "COMMAND"`
### bash起動
- `docker exec -it eppstein bash`
- `-i`は標準入力を開き続けるため，`-t`はttyを割り当てるため
### bash切断
- `exit`


