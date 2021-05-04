#!/bin/bash

INC_MAX_SCALE=27     # Max scale to test (inclusive).
DEGREE=16            # Degree as per Graph500 specs.

genExe=graphsize.exe # Graph generator.
logFile=run.log      # File for graphsize output.
sizeFile=size.log    # File for each graph size.

# Clear log files.
rm $logFile; touch $logFile 
rm $sizeFile; touch $sizeFile

for (( scale=1; scale <= $INC_MAX_SCALE; scale++ ))
do
    gFile="graph_$scale.wsg"
    
    # Generate graph.
    echo $gFile >> $logFile
    ./$genExe -g $scale -k $DEGREE -b $gFile >> $logFile 2>&1

    echo $gFile >> $sizeFile
    stat -c %s $gFile >> $sizeFile

    rm $gFile
done

