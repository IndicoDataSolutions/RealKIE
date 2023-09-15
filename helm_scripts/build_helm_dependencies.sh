#!/bin/bash

set -e
fullpath=$1
dir=$(dirname $fullpath)
echo "helm-dependency-fetch $dir"
name=$(helm show chart $dir | yq '.name')
for dep in $(helm dependency list $dir | grep 'file://' | awk '{print $1}')
do
  echo "  [dependency]: helm-dependency-fetch $dep"
  set +e
  helm-dependency-fetch $dep >> /dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo "Re-doing $dir with helm dependency build"
    helm dependency build $dir >> /dev/null 2>&1
  fi
  set -e
done
set +e
helm-dependency-fetch $dir >> /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Re-doing $dir with helm dependency build"
  helm dependency build $dir >> /dev/null 2>&1
fi
set -e
