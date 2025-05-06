#!/usr/bin/env bash
set -x
set -e

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

cp requirements.txt submit/
cp requirements_dev.txt submit/
