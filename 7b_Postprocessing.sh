#!/bin/bash

prjname=$1
file=$2

OutFileName=$prjname/output/Predictions/$file.bed
for filename in ./$prjname/output/Predictions/Chr*_$file.bed; do
 if [ "$filename"  != "$OutFileName" ] ;
 then 
   cat "$filename" >>  "$OutFileName" # Append 
 fi 
done
