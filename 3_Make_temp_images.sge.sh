#!/bin/bash
# job scheduler may be inserted here

echo "3_Make_temp_images"

prjname=$1
leftout_chr=$2
colorscheme=$3
leftminusone=$((leftout_chr-1))
leftplusone=$((leftout_chr+1))

echo $prjname
echo $leftout_chr

for chr in $(seq 1 22); do
if [ $chr -ne $leftout_chr ]
then
    mkdir -p $prjname/Temp/leave_out_$leftout_chr/test_on_$chr/Test/Positive/
    mkdir -p $prjname/Temp/leave_out_$leftout_chr/test_on_$chr/Test/Negative/
    mkdir -p $prjname/Temp/leave_out_$leftout_chr/test_on_$chr/Train/Positive/
    mkdir -p $prjname/Temp/leave_out_$leftout_chr/test_on_$chr/Train/Negative/

    #copy all test data:
    cp $prjname/All_Images/$colorscheme/Positive/chr$chr/* $prjname/Temp/leave_out_$leftout_chr/test_on_$chr/Test/Positive/
    cp $prjname/All_Images/$colorscheme/Negative/chr$chr/* $prjname/Temp/leave_out_$leftout_chr/test_on_$chr/Test/Negative/

    #copy all train data:
    for chr2 in $(seq 1 22); do
	if [ $chr2 -ne $leftout_chr ]
	then
	if [ $chr2 -ne $chr ]
	then
	cp $prjname/All_Images/$colorscheme/Positive/chr$chr2/* $prjname/Temp/leave_out_$leftout_chr/test_on_$chr/Train/Positive/
	cp $prjname/All_Images/$colorscheme/Negative/chr$chr2/* $prjname/Temp/leave_out_$leftout_chr/test_on_$chr/Train/Negative/
	fi
	fi
    done
fi
done
