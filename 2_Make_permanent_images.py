import numpy as np
from numpy import zeros, newaxis
import fileinput
from PIL import Image
import math
import csv    
import os
import pandas as pd
import sys
import gzip

#order from https://www.nature.com/articles/nature14248/figures/2
#old order:
order_thymus_moved = [17,         2, 8, 1, 15, 14, 16, 3, 24,        20, 19, 18, 21, 22,        7, 9, 10, 13, 12, 11, 4, 5, 6,        62, 34, 45, 33, 44, 43, 39, 41, 42, 40, 37, 48, 38, 47,    112, 93,     29, 31, 35, 51, 50, 36, 32, 46, 30,         26, 49, 25, 23,         52,         55, 56, 59, 61, 57, 58, 28, 27,         54, 53,                  71, 74, 68, 69, 72, 67, 73, 70, 82, 81,         63,         100, 108, 107, 89, 90,         83, 104, 95, 105, 65,         78, 76, 103, 111,         92, 85, 84, 109, 106, 75, 101, 102, 110, 77, 79, 94,         99, 86, 88, 97, 87, 80, 91, 66, 98, 96, 113]
#new_order:
order_IS_Block = [17,         2, 8, 1, 15, 14, 16, 3, 24,        20, 19, 18, 21, 22,        7, 9, 10, 13, 12, 11, 4, 5, 6,          29, 31, 35, 32, 46, 30,         62, 34, 45, 33, 44, 43, 39, 41, 42, 40, 37, 48, 38, 47,    112, 93,     51, 50, 36,         113,         75, 101, 102, 92, 85, 84, 109, 106, 110, 77, 79, 94,          26, 49, 25, 23,         52,         55, 56, 59, 61, 57, 58, 28, 27,         54, 53,                  71, 74, 68, 69, 72, 67, 73, 70, 82, 81,         63,         100, 108, 107, 89, 90,         83, 104, 95, 105, 65,         78, 76, 103, 111,        99, 86, 88, 97, 87, 80, 91, 66, 98, 96]
order_Tcells = [17,         2, 8, 1, 15, 14, 16, 3, 24,        20, 19, 18, 21, 22,        7, 9, 10, 13, 12, 11, 4, 5, 6,        62, 34, 33, 39, 40, 43, 45,  44, 42, 41, 37, 38, 48, 47,    112, 93,     29, 31, 35, 51, 50, 36, 32, 46, 30,         26, 49, 25, 23,         52,         55, 56, 59, 61, 57, 58, 28, 27,         54, 53,                  71, 74, 68, 69, 72, 67, 73, 70, 82, 81,         63,         100, 108, 107, 89, 90,         83, 104, 95, 105, 65,         78, 76, 103, 111,         92, 85, 84, 109, 106, 75, 101, 102, 110, 77, 79, 94,         99, 86, 88, 97, 87, 80, 91, 66, 98, 96, 113]

order20 = [17, 2, 8, 1, 15, 14, 16, 3, 20, 19, 18, 7, 9, 10, 13, 12, 11, 4, 5, 6]
chrom_lengths = {"chr1":248956422, "chr2":242193529, "chr3":198295559, "chr4":190214555, "chr5":181538259, "chr6":170805979, "chr7":159345973, "chrX":156040895, "chr8":145138636, "chr9":138394717, "chr11":135086622, "chr10":133797422, "chr12":133275309, "chr13":114364328, "chr14":107043718, "chr15":101991189, "chr16":90338345, "chr17":83257441, "chr18":80373285, "chr20":64444167, "chr19":58617616, "chrY":57227415, "chr22":50818468, "chr21":46709983}

#------------------

prjname=sys.argv[1]
image_length=int(sys.argv[2])
colorscheme=sys.argv[3]

print(prjname)
print(image_length)
print(colorscheme)

NegOrPos=sys.argv[4] 
if NegOrPos == "Negative":
    CSV_file=prjname + "/Negative_samples_by_chr.csv"
if NegOrPos == "Positive":
    CSV_file=prjname + "/cat1+2+Diff_preprocessed.csv"

print(NegOrPos)
print(CSV_file)

#--------------------
#Defining Color Schemes: Color, BW, Grey:

def colourState(a):
#define colours for 15 different states
#roadmap colours

    if a==15:
        return [255, 255, 255]
    elif a==14:
        return [192, 192, 192]
    elif a==13:
        return [128, 128, 128]
    elif a==12:
        return [189, 183, 107]
    elif a==11: #BivFlnk
        return [50, 205, 50]
    elif a==10: #TssBiv
        return [205, 92, 92]
    elif a==9:
        return [138, 145, 208]
    elif a==8:
        return [102, 205, 170]
    elif a==7:
        return [255, 255, 0]
    elif a==6:
        return [194, 225, 5]
    elif a==5: #TxWk
    #    return [0, 100, 0]
        return [50, 205, 50]
    elif a==4: #Tx
    #    return [0, 128, 0]
        return [0, 100, 0]
    elif a==3: #TxFlnk
    #    return [50, 205, 50]
        return [0, 128, 0]
    elif a==2:
        return [255, 69, 0]
    elif a==1:
        return [255, 0, 0]
    elif a=='\n':
        return '\r\n'

def colourStateBW(a):
#define colours for 15 different states
#roadmap colours

    if a==15:
        return [255, 255, 255]
    elif a==14:
        return [255, 255, 255]
    elif a==13:
        return [255, 255, 255]
    elif a==12:
        return [255, 255, 255]
    elif a==11: #BivFlnk
        return [255, 255, 255]
    elif a==10: #TssBiv
        return [255, 255, 255]
    elif a==9:
        return [255, 255, 255]
    elif a==8:
        return [0, 0, 0]
    elif a==7:
        return [0, 0, 0]
    elif a==6:
        return [0, 0, 0]
    elif a==5: #TxWk
    #    return [0, 100, 0]
        return [0, 0, 0]
    elif a==4: #Tx
    #    return [0, 128, 0]
        return [0, 0, 0]
    elif a==3: #TxFlnk
    #    return [50, 205, 50]
        return [0, 0, 0]
    elif a==2:
        return [0, 0, 0]
    elif a==1:
        return [0, 0, 0]
    elif a=='\n':
        return '\r\n'

def colourStateGrey(a):
#define colours for 15 different states
#roadmap colours

    if a==15:
        return [255, 255, 255]
    elif a==14:
        return [242.25, 242.25, 242.25]
    elif a==13:
        return [242.25, 242.25, 242.25]
    elif a==12:
        return [204, 204, 204]
    elif a==11: #BivFlnk
        return [229.5, 229.5, 229.5]
    elif a==10: #TssBiv
        return [216.75, 216.75, 216.75]
    elif a==9:
        return [242.25, 242.25, 242.25]
    elif a==8:
        return [178, 178, 178]
    elif a==7:
        return [102, 102, 102]
    elif a==6:
        return [127.5, 127.5, 127.5]
    elif a==5: #TxWk
    #    return [0, 100, 0]
        return [153, 153, 153]
    elif a==4: #Tx
    #    return [0, 128, 0]
        return [51, 51, 51]
    elif a==3: #TxFlnk
    #    return [50, 205, 50]
        return [76.5, 76.5, 76.5]
    elif a==2:
        return [25.5, 25.5, 25.5]
    elif a==1:
        return [0, 0, 0]
    elif a=='\n':
        return '\r\n'


#------------------
#Define Order of Rows:
def sort_rows(inputarray, old_order, new_order):
    your_permutation = [new_order.index(i) for i in old_order ]
    idx = np.empty_like(your_permutation)
    idx[your_permutation] = np.arange(len(your_permutation))
    outputarray = inputarray[idx, :] 
    return outputarray

#-----------------
#Function to make images (start of gene is in middle of picture):

#make png image
def make_image(chromosome, gene_start, gene_stop, image_length, n_datasets=111, pm="+", posneg="Negative", color="Color"):
    
    '''
    Args:
        chromosome: chromosome in format "chr1"
        gene_start: bp number of gene start
        gene_stop: bp number of gene stop
        image_length: bps of image
        n_datasets: number of considered datasets. If not for test purposes(20) it should be 111.
        minus: whether the gene is on the "+"/"-" strang       
    '''
    #Gene Start at center of image
    

    if not os.path.exists(prjname + '/All_Images/' + color + '/' + posneg + '/%s' % chromosome +'/'):
        os.makedirs(prjname + '/All_Images/' + color + '/' + posneg + '/%s' % chromosome +'/')

    
    #make images:
    w, h = int(image_length/200), n_datasets
    data = np.zeros((h, w, 3), dtype=np.uint8)
    
    if pm == "-":
        start_pos = math.trunc(gene_stop/200-image_length/400)    #data saved in steps of 200 bps.
        if start_pos < 0:
            start_pos = 0
        end_pos = start_pos + int(image_length/200)

        with gzip.open('../Data/chr-new-order/%s_new-order.csv.gz' % chromosome, 'r') as inp:
            k=0
            for row in csv.reader(inp, delimiter=';'):
                k+=1
                for j in range(w):
                    if color=="Color":
                        data[k-1, j] = colourState(int(float(row[j+start_pos])))
                    if color=="BW":
                        data[k-1, j] = colourStateBW(int(float(row[j+start_pos])))
                    if color=="Grey":
                        data[k-1, j] = colourStateGrey(int(float(row[j+start_pos])))
    
        #data = sort_rows(data, order_thymus_moved, order_IS_Block)
        #data = sort_rows(data, order_thymus_moved, order_Tcells)
        img = Image.fromarray(data, 'RGB')
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img.save(prjname + '/All_Images/' + color + '/' + posneg + '/%s' % chromosome + '/%s' % chromosome + '_%d' % gene_start + '_%d' % gene_stop + '_mirrored.png')
        
    else: 
        start_pos = math.trunc(gene_start/200-image_length/400)    #data saved in steps of 200 bps.
        if start_pos < 0:
            start_pos = 0
        end_pos = start_pos + int(image_length/200)

        with gzip.open('../Data/chr-new-order/%s_new-order.csv.gz' % chromosome, 'r') as inp:
            k=0
            for row in csv.reader(inp, delimiter=';'):
                k+=1
                for j in range(w):
                    if color=="Color":
                        data[k-1, j] = colourState(int(float(row[j+start_pos])))
                    if color=="BW":
                        data[k-1, j] = colourStateBW(int(float(row[j+start_pos])))
                    if color=="Grey":
                        data[k-1, j] = colourStateGrey(int(float(row[j+start_pos])))  
                    
        #data = sort_rows(data, order_thymus_moved, order_IS_Block)
        #data = sort_rows(data, order_thymus_moved, order_Tcells)
        img = Image.fromarray(data, 'RGB')            
        img.save(prjname + '/All_Images/' + color + '/' + posneg + '/%s' % chromosome +'/%s' % chromosome + '_%d' % gene_start + '_%d' % gene_stop + '.png')
    #img.show()  

#-------------------------------------------------------------------
#Making Images:

df = pd.read_csv(CSV_file, header=None, sep='\t')

for i in range(len(df[0])):
    print (i)
    print (df[0].iloc[i], df[1].iloc[i], df[2].iloc[i])
    make_image(df[0].iloc[i], df[1].iloc[i], df[2].iloc[i], image_length, 111, df[3].iloc[i], NegOrPos, colorscheme)
    
print('done')
