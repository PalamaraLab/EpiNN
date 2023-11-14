#!/bin/bash

prjname=$1

#OutFileName=$prjname/output/CNNs_Output.csv
OutFileName=$prjname/bootstrap/all.csv 
for filename in $prjname/bootstrap/*.csv; do 
 if [ $filename  != $OutFileName ] ;      # Avoid recursion 
 then 
 echo $filename
 cat  $filename >>  $OutFileName # Append from the 2nd line each file 
 fi
done

