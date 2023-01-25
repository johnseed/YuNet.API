#!/bin/bash

# enable conda for this shell
. /opt/conda/etc/profile.d/conda.sh

# activate the environment
conda activate face
# source face/bin/activate
# exec the cmd/command in this process, making it pid 1
exec "$@"
# exec "cd /app && uvicorn --host 0.0.0.0 --port 8000 fastapi-server:app"