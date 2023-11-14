#!/bin/bash

prjname=$1

rm -f $prjname/output/CNNs_weigthed_average.csv
rm -f $prjname/output/jackknife.txt


AverageFileName=$prjname/output/CNNs_weigthed_average.csv

for filename in $prjname/output/CNNs_leaveout_*.csv; do
#    echo $filename
    totweight=( $( awk '{totweight+=$3}END{print totweight}' $filename ) )
    awk '{train+=$2; test+=$3; pr+=$4*$3/'$totweight'; acc+=$5*$3/'$totweight'; prec+=$6*$3/'$totweight'; rec+=$7*$3/'$totweight'} END {print "'$filename'", train, test ,pr, acc, prec, rec}' $filename >> $AverageFileName
done

sort -nk1 -o $AverageFileName $AverageFileName


#Jackknife
JackknifeFileName=$prjname/output/CNNs_Jackknife.csv

#PRAUC
echo -ne "PRAUC \t" >> $prjname/output/jackknife.txt
cat $AverageFileName | awk 'NR==1 {print $4}' > $prjname/output/temp.estimatorForAll
cat $AverageFileName | awk 'NR>1 {print $4}'  > $prjname/output/temp.valuesFile
cat $AverageFileName | awk 'NR>1 {print $3}'  > $prjname/output/temp.weightsFile
sh weightedJackKnife.sh $prjname/output/temp.valuesFile $prjname/output/temp.weightsFile $prjname/output/temp.estimatorForAll | tr '\n' '\t'  >> $prjname/output/jackknife.txt
echo -ne '\n' >> $prjname/output/jackknife.txt

#ACC
echo -ne "ACC \t" >> $prjname/output/jackknife.txt
cat $AverageFileName | awk 'NR==1 {print $5}' > $prjname/output/temp.estimatorForAll
cat $AverageFileName | awk 'NR>1 {print $5}'  > $prjname/output/temp.valuesFile
cat $AverageFileName | awk 'NR>1 {print $3}'  > $prjname/output/temp.weightsFile
sh weightedJackKnife.sh $prjname/output/temp.valuesFile $prjname/output/temp.weightsFile $prjname/output/temp.estimatorForAll | tr '\n' '\t'  >> $prjname/output/jackknife.txt
echo -ne '\n' >> $prjname/output/jackknife.txt

#Prec
echo -ne "Prec \t" >> $prjname/output/jackknife.txt
cat $AverageFileName | awk 'NR==1 {print $6}' > $prjname/output/temp.estimatorForAll
cat $AverageFileName | awk 'NR>1 {print $6}'  > $prjname/output/temp.valuesFile
cat $AverageFileName | awk 'NR>1 {print $3}'  > $prjname/output/temp.weightsFile
sh weightedJackKnife.sh $prjname/output/temp.valuesFile $prjname/output/temp.weightsFile $prjname/output/temp.estimatorForAll | tr '\n' '\t'  >> $prjname/output/jackknife.txt
echo -ne '\n' >> $prjname/output/jackknife.txt

#Rec
echo -ne "Rec \t" >> $prjname/output/jackknife.txt
cat $AverageFileName | awk 'NR==1 {print $7}' > $prjname/output/temp.estimatorForAll
cat $AverageFileName | awk 'NR>1 {print $7}'  > $prjname/output/temp.valuesFile
cat $AverageFileName | awk 'NR>1 {print $3}'  > $prjname/output/temp.weightsFile
sh weightedJackKnife.sh $prjname/output/temp.valuesFile $prjname/output/temp.weightsFile $prjname/output/temp.estima\
torForAll | tr '\n' '\t'  >> $prjname/output/jackknife.txt
echo -ne '\n' >> $prjname/output/jackknife.txt
