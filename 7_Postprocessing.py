import pandas as pd
import numpy as np
import copy
import sys

prjname=str(sys.argv[1])
imagesize = int(sys.argv[2])
stepsize = int(sys.argv[3])
chromosome = str(sys.argv[4])

number_steps=int(imagesize/stepsize)

print("prjname:" + prjname)
print("imagesize:" + str(imagesize))
print("stepsize:" + str(stepsize))

def read_chr_predictions(chr):   
    chrDF = pd.read_csv(prjname + "/output/Predictions/Predictions_raw_chr"+ str(chr) +".csv", usecols=[2,3,4,5], delimiter='\t')

    #Add rows at end
    while chrDF["Start"].iloc[-1] < chrDF["End"].iloc[-1]:
        chrDF = chrDF.append(chrDF.tail(1))
        chrDF["Start"].iloc[-1]=chrDF["Start"].iloc[-1]+stepsize

    #Make average column:
    chrDF["Forward-Gaussian-Average"]=chrDF["Forward"].rolling(number_steps, min_periods=1, win_type='gaussian').mean(std=0.5)
    chrDF["Backward-Gaussian-Average"]=chrDF["Backward"].rolling(number_steps, min_periods=1, win_type='gaussian').mean(std=0.5)
    chrDF["Max-Moving-Average"]=chrDF[["Forward-Gaussian-Average", "Backward-Gaussian-Average"]].max(axis=1)

    #Make Columns into annotation format:
    chrDF["End"]=chrDF["Start"]+stepsize
    chrDF = chrDF.drop(["Forward", "Backward"], axis=1)
    chrDF["Chromosome"]="chr"+str(chr)
    chrDF["PM"]=chrDF[["Forward-Gaussian-Average", "Backward-Gaussian-Average"]].apply(lambda x: "+" if x["Forward-Gaussian-Average"]>x["Backward-Gaussian-Average"] else "-", axis=1)
    chrDF["Name"]="NA"
    chrDF=chrDF[["Chromosome", "Start", "End", "Name", "Max-Moving-Average","PM"]]
    return chrDF

#Save Gene Annotation .bed file
allchrDF=pd.DataFrame()
allchrDF = allchrDF.append(read_chr_predictions(chromosome))
allchrDF.to_csv(prjname + "/output/Predictions/Chr" + chromosome + "_Genome-Annotation.bed",header=None, index=False, sep="\t")

#####Gene Mapping#######
allchrDF=pd.read_csv(prjname + "/output/Predictions/Chr" + chromosome + "_Genome-Annotation.bed",header=None, delimiter="\t")
genelistchr = pd.read_csv(r'../Data/hg38_refSeq_genes_HUGO.bed', header=None, delimiter='\t')
genelistchr[0]=genelistchr[0].str[3:]
genelistchr = genelistchr.sort_values(by=[0,1,2])
genelistchr[0]="chr" + genelistchr[0]

#print(allchrDF.head())

combinedList=[]

allchrDF=pd.read_csv(prjname + "/output/Predictions/Chr" + chromosome + "_Genome-Annotation.bed",header=None, delimiter="\t")
genelistchr = pd.read_csv(r'../Data/hg38_refSeq_genes_HUGO.bed', header=None, delimiter='\t')
genelistchr[0]=genelistchr[0].str[3:]
genelistchr = genelistchr.sort_values(by=[0,1,2])
genelistchr[0]="chr" + genelistchr[0]


genelistchr=genelistchr.loc[genelistchr[0] == "chr" + str(chromosome)]
allchrDF=allchrDF.loc[allchrDF[0] == "chr" + str(chromosome)]
active_list=[]
k=0
for i in range(len(allchrDF[1])):     
    start = allchrDF[1].iloc[i]
    end = allchrDF[2].iloc[i]
    
    #update current active list
    active_list = [x for x in active_list if x[2] > start]
    for x in range(0,len(active_list)):
        active_list[x][4]=allchrDF[4].iloc[i]
       
    #update active list with new genes
    for j in range(k,len(genelistchr[1])):
        if genelistchr[1].iloc[j]>end:
            break     
        k=j+1
            
        genelistchr[4].iloc[j] = allchrDF[4].iloc[i]
        active_list.append(genelistchr.iloc[j])
    #combinedList.extend(active_list)
    combinedList.extend(copy.deepcopy(active_list))
print (chromosome)

#Group Genes by name (not isoform specific)
GeneDF=pd.DataFrame(combinedList)    
GeneDF = GeneDF.groupby([3,0,5],as_index=False, sort=False).max()
GeneDF = GeneDF[[0,1,2,3,4,5]]
GeneDF.to_csv(prjname + "/output/Predictions/Chr"+ chromosome +"_predictions-HUGO.bed", index=False, header=None, sep='\t')   




#subtract Training List and BMLS List
#Training List:
trainsetDF = pd.read_csv("../Data/cat1+2+Diff_Names.csv", header=None, delimiter='\t')
#BMLS List:
BMLSsetDF = pd.read_csv("../Data/HPA_BMLS_849_genes.csv", header=None, delimiter='\t')


#predictions without training data
noTrainDF = pd.DataFrame()
for i in range(len(GeneDF[0])):
    p=0
    for j in range(len(trainsetDF[0])):
        if trainsetDF[0].iloc[j] == GeneDF[3].iloc[i]:
            p=1        
    if p==0:
        noTrainDF = noTrainDF.append(GeneDF.iloc[i], ignore_index = True)
noTrainDF.to_csv(prjname + "/output/Predictions/Chr"+ chromosome +"_no-train.bed", index=False, header=None, sep='\t')  
#print(noTrainDF.head())

#predictions without training data, without BMLS gene set
noTrainBMLSDF = pd.DataFrame()
for i in range(len(noTrainDF[0])):
    p=0
    for j in range(len(BMLSsetDF[0])):
        if BMLSsetDF[0].iloc[j] == noTrainDF[3].iloc[i]:
            p=1        
    if p==0:
        noTrainBMLSDF = noTrainBMLSDF.append(noTrainDF.iloc[i], ignore_index = True)
noTrainBMLSDF.to_csv(prjname + "/output/Predictions/Chr"+ chromosome +"_no-train-BMLS.bed", index=False, header=None, sep='\t') 
#print(noTrainBMLSDF.head())
