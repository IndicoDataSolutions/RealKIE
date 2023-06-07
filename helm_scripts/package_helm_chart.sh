#!/bin/bash

set -e
fullpath=$1
BRANCH_NAME=$2

devel=''
dir=$(dirname $fullpath)

name=$(helm show chart $dir | yq '.name')
version=$(helm show chart $dir | yq '.version')
harbor='harborprod'

echo  "Fullpath: ${dir}, Branch: ${BRANCH_NAME}, chart ${name}"
echo ""
echo "------------------------------------------------------------------"
# not building on main
if [ ! -z "$BRANCH_NAME" ]; then
  branch=${BRANCH_NAME//\//\-} # replace slashes with -
  branch=${branch//_/\-} # replace underscores with -
  
  if [ ! -z "$DRONE_TAG" ]; then
    version=$version-${DRONE_TAG}
  else
    version=$version-$branch
  fi
  harbor='harbordev'
  devel='--devel'
else
# on main, respect the tag if it exists
  if [ ! -z "$DRONE_TAG" ]; then
    version=$version-${DRONE_TAG}
  fi
fi

num_charts=$((num_charts+1))

echo "Working on Chart $name, Version: $version"

if [ -d "$dir/tests" ]
then
  for testfile in $(find $dir/tests -name '*.yaml')
  do
    testname=$(basename "$testfile")
    echo "Running test with $testfile"
  
    echo helm template ./$dir --dependency-update --name-template $testname --namespace default --kube-version 1.20 --values $testfile --include-crds --debug > /dev/null
    helm template ./$dir --dependency-update --name-template $testname --namespace default --kube-version 1.20 --values $testfile --include-crds --debug > /dev/null
    
    echo "Linting chart"
    helm lint ./$dir --values $testfile
  done
fi

echo "helm package $dir -d $dir --version $version"
helm package $dir -d $dir --version $version
