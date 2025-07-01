#!/bin/bash
if command -v wget >/dev/null 2>&1; then
  wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000356/data/data.tar.gz
else
  curl -O https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000356/data/data.tar.gz
fi
tar -zxvf data.tar.gz
