#!/bin/bash

param=$1

#  Run the program:
#      - read the common inputfile(s)
#      - save output in specific outputfile
python3 /home/hlm/testfolder/basinHoppingWithParse.py $param  # > ${param}.out

# - Copy back the outputfile
cp -r * $PBS_O_WORKDIR/data/${PBS_JOBID}

##!/bin/sh
