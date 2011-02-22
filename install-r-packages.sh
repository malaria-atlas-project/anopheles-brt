#!/bin/bash

wget http://cran.r-project.org/src/contrib/gbm_1.6-3.1.tar.gz
tar -xvf gbm_1.6-3.1.tar.gz
R CMD INSTALL gbm
