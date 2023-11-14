import numpy as np
import pandas as pd
import csv
import sys
pd.options.mode.chained_assignment = None

chrom_lengths = {"chr1":248956422, "chr2":242193529, "chr3":198295559, "chr4":190214555, "chr5":181538259, "chr6":170805979, "chr7":159345973, "chrX":156040895, "chr8":145138636, "chr9":138394717, "chr11":135086622, "chr10":133797422, "chr12":133275309, "chr13":114364328, "chr14":107043718, "chr15":101991189, "chr16":90338345, "chr17":83257441, "chr18":80373285, "chr20":64444167, "chr19":58617616, "chrY":57227415, "chr22":50818468, "chr21":46709983}

prjname = sys.argv[1]
image_length = int(sys.argv[2])

print (prjname)
print (image_length)

#Positives
#Read original .bed file of positives
df = pd.read_csv(r'../Data/cat1+2+Diff.bed', header=None, delimiter='\t')
#Remove all genes that are too close to the chromosome end: start+image_length >= chr-end. 
def process_row(x):
    return True if x[1]+2*image_length < chrom_lengths[x[0]] else False
df = df[df.apply(lambda row: process_row(row), axis=1)]
#Remove all genes that are too close to the chromosome start: start-image_length <0.
def process_row_start(x):
    return True if x[1]-2*image_length > 0 else False
df = df[df.apply(lambda row: process_row_start(row), axis=1)]
#Drop Columns: Name, Value; Leave colmns: chr, start, end, strand
df = df.drop(df.columns[[3,4]], axis=1)
#Save as processed .csv file, to be used to make positive images:
df.to_csv(prjname + '/cat1+2+Diff_preprocessed.csv', sep ="\t", header=None, index=False)


#Negatives
#Read original file of all genes:
dfn = pd.read_csv(r'../Data/Negatives.csv', header=None, delimiter=';')
#drop value column
dfn = dfn.drop(dfn.columns[[4]], axis=1)
#map +1 and -1 to +/-
di = {1: "+", -1: "-"}
dfn[3].replace(di, inplace=True)
dfn[0]="chr" + dfn[0].astype(str)
#Remove all genes that are too close to the chromosome end: start+image_length >= chr-end.
def process_row(x):
    return True if x[1]+2*image_length < chrom_lengths[x[0]] else False
dfn = dfn[dfn.apply(lambda row: process_row(row), axis=1)]
#Remove all genes that are too close to the chromosome start: start-image_length <0.
def process_row_start(x):
    return True if x[1]-2*image_length > 0 else False
dfn = dfn[dfn.apply(lambda row: process_row_start(row), axis=1)]
#remove genes close to positive genes:
merge_df = dfn.merge(df, on=0, how='left')
#make positive genes "bigger" by adding image length in both directions
#negative start between positive start and end
merge_df.drop(merge_df[(merge_df['1_x'] >= merge_df['1_y']-image_length) & (merge_df['1_x'] <= merge_df['2_y']+image_length)].index , inplace=True)
#negative end between positive start and end
merge_df.drop(merge_df[(merge_df['2_x'] >= merge_df['1_y']-image_length) & (merge_df['2_x'] <= merge_df['2_y']+image_length)].index , inplace=True)
#positive gene fully in negative gene
merge_df.drop(merge_df[(merge_df['1_x'] <= merge_df['1_y']-image_length) & (merge_df['2_x'] >= merge_df['2_y']+image_length)].index , inplace=True)
#negative gene fully in positive gene
merge_df.drop(merge_df[(merge_df['1_x'] >= merge_df['1_y']-image_length) & (merge_df['2_x'] <= merge_df['2_y']+image_length)].index , inplace=True)
merge_df.drop(['1_y', '2_y', 5], axis=1, inplace=True)
dfn=merge_df

#delete duplicates
dfn = dfn.drop_duplicates(keep='first')

#Save as processed .csv file, to be used to make positive images:
dfn.to_csv(prjname + '/Negative_preprocessed.csv', sep ="\t", header=None, index=False)

#match numbers within each chromosome
#output: negative list, matching chromosome numbers
df_negative_sample = pd.DataFrame()
for i in range(1,23):
    dfp_chr = df[df[0]=='chr%d' %i]
    dfp_chr_rows = dfp_chr.shape[0]
    dfn_chr = dfn[dfn[0]=='chr%d' %i]
    dfn_small = dfn_chr.sample(n=dfp_chr_rows, random_state=123)
    df_negative_sample= df_negative_sample.append(dfn_small, ignore_index=True)  
for i in ('X'):
    #for 1 chromosome:
    dfp_chr = df[df[0]=='chr%s' %i]
    dfp_chr_rows = dfp_chr.shape[0]
    dfn_chr = dfn[dfn[0]=='chr%s' %i]
    dfn_small = dfn_chr.sample(n=dfp_chr_rows, random_state=123)
    df_negative_sample= df_negative_sample.append(dfn_small)    
df_negative_sample.to_csv(prjname + '/Negative_samples_by_chr.csv', sep ="\t", header=None, index=False)
