#!/bin/bash

set -e

# Requirements:
# 1. harbor.devops.indico.io is already added as a helm repo (named harbor)
# 2. cm-push is already installed as a helm plugin
# 3. yq has been added to the container

echo "Fetching Chart Dependencies"
helm_scripts/build_helm_dependencies.sh ./charts/Chart.yaml
echo "Finished Dependencies"

echo "Packaging Charts"
helm_scripts/package_helm_chart.sh ./charts/Chart.yaml "$1"
echo "Finished Packaging"

echo "Pushing Charts"
helm_scripts/push_helm_chart.sh ./charts/Chart.yaml "$1"
echo "Finished Chart Uploads"
