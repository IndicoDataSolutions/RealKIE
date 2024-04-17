#!/bin/bash

set -e
fullpath=$1
BRANCH_NAME=$2

devel=''
dir=$(dirname $fullpath)
name=$(helm show chart $dir | yq '.name')
version=$(helm show chart $dir | yq '.version')
harbor='harborprod'
devel=''

echo ""
echo "-------------------------------------------------------------------------------------------"
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

    echo "Images referenced"
    helm template ./$dir --dependency-update --name-template $testname --namespace default --kube-version 1.20 --values $testfile --include-crds | yq '..|.image? | select(.)' | sort -u
  done
fi

#helm dependency build ./$dir
#helm package ./$dir -d ./$dir --version $version

#Push chart, check if it succeeded, if not, retry.
pushed="false"
retry_attempts=10
push_flags="--force"

until [ $pushed == "true" ] || [ $retry_attempts -le 0 ]
do
  chart_name="harborprod/${name}:${version}"
  #Verifying pushed chart.
  if helm_scripts/validate_chart.sh "${name}" "${version}"; then
    echo "Successfully uploaded ${chart_name}"
    pushed="true"
  else
    if [ $retry_attempts -ne 10 ]; then
      push_flags="--force"
      echo "Retry push ${chart_name} [$retry_attempts] Sleep 10"
      sleep 10
    else
      echo "Pushing chart ${chart_name}"
    fi
    set +e
    echo "helm cm-push $dir --version $version harborprod ${push_flags} [$retry_attempts]"
    helm cm-push $dir --version $version harborprod ${push_flags} 
    set -e
    pushed="false"
    ((retry_attempts--))
  fi
done

set -e
# double-check that the chart was pushed.
if [ $retry_attempts -le 0 ]; then
  echo "Error: Unable to push $harborprod/${name}:${version}"
  exit 1
fi
