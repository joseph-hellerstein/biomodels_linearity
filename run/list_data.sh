#!/bin/bash
ls -l data/piecewise*max*.csv |sed 's/^.*maxreduction_//' | sort | sed 's/__many.*repeat//'
