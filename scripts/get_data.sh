#!/bin/bash
aws s3 sync s3://project-fruitfly /datasets --endpoint-url=https://s3.us-east-2.wasabisys.com --no-sign-request