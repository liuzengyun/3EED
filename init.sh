#!/bin/bash

# compile custom operators

cd ops/teed_pointnet/pointnet2_batch
pip install -e . --no-build-isolation

cd ../roiaware_pool3d
pip install -e . --no-build-isolation

cd ../../../pointnet2
pip install -e . --no-build-isolation
cd ../../..

