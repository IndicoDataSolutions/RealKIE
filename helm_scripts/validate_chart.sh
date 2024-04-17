#!/bin/bash
set -e 

repo=$1
version=$2

curl -f -s -X 'GET' \
"https://harbor.devops.indico.io/api/v2.0/chartrepo/indico-charts/charts/${repo}/${version}/labels" \
-H "accept: application/json" \
-H "authorization: Basic ${HARBOR_API_TOKEN}" \
> /dev/null
