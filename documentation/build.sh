#!/bin/sh

set -e

if [ "$(basename $(pwd))" != "documentation" ]; then
  echo "Please run from richard/documentation directory"
  exit 1
fi

mkdir -p ./build && cd ./build
pdflatex ../src/math.tex
mv ./math.pdf ../
