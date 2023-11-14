import numpy as np
import csv
import math
import keras
from keras.models import load_model
import scipy
import tensorflow as tf
from PIL import Image
import pandas as pd
import sys
import gc


prjname=sys.argv[1]
width=int(sys.argv[2])
stepsize=int(sys.argv[3])
colorstate=sys.argv[4]
image_size=int(sys.argv[5])
run_on_chr=str(sys.argv[6])


print (prjname)
print(width) 
print (stepsize)
print(colorstate)
print (image_size)


# MODEL
model = load_model(prjname + '/output/'+ prjname +'.h5')



chrom_lengths = {"chr1":248956422, "chr2":242193529, "chr3":198295559, "chr4":190214555, "chr5":181538259, "chr6":170805979, "chr7":159345973, "chrX":156040895, "chr8":145138636, "chr9":138394717, "chr11":135086622, "chr10":133797422, "chr12":133275309, "chr13":114364328, "chr14":107043718, "chr15":101991189, "chr16":90338345, "chr17":83257441, "chr18":80373285, "chr20":64444167, "chr19":58617616, "chrY":57227415, "chr22":50818468, "chr21":46709983}


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



def make_image_faster(data, chromosome, gene_start, gene_stop, image_length, n_datasets=111):
    '''
    Args:
        chromosome: name of chromosome in format "1"
        gene_start: bp number of gene start
        gene_stop: bp number of gene stop
        n_datasets: number of considered datasets. If not for test purposes(20) it should be 111.
    '''
    test = 0
    chromosome = 'chr%s' %chromosome

    if gene_start > chrom_lengths[chromosome]:
        test = 1
    start_pos = math.trunc(gene_start/200)
#    start_pos = math.trunc(gene_stop/200-image_length/400)    #data saved in steps of 200 bps.
    if start_pos < 0:
        start_pos = 0
    end_pos = start_pos + int(image_length/200)
    if test == 0:
        img = Image.fromarray(data[:,start_pos:end_pos], 'RGB')
    return img

def load_chr_data(chromosome, colorstate):
    chromosome = 'chr%s' %chromosome
    print("Now Loading Data")

    dataDF = pd.read_csv('../Data/chr-new-order/%s_new-order_transpose.csv' % chromosome,  sep=',', header=None, dtype=np.uint8)

    data = dataDF.to_numpy().transpose()
#    print ("dataDF " + str(sys.getsizeof(dataDF)))
#    print ("data " + str(sys.getsizeof(data)))

    del dataDF
    gc.collect()

    print ('Loaded Data, now apply color transformation:')

    data2= np.zeros((111, len(data[1]), 3), dtype=np.uint8)
    for k in range(111):
        for j in range(len(data[1])):
            if colorstate == "BW":
                #data2[k,j] = np.uint8(colourStateBW(int(data[k,j])))
                data2[k,j] = colourStateBW(int(data[k,j]))
            elif colorstate == "Color":
                data2[k,j] = colourState(int(data[k,j]))
            elif colorstate == "Grey":
                data2[k,j] = colourStateGrey(int(data[k,j]))
            else:
                print ('WARNING! Colorstate undefined')
#    print ("data2 " + str(sys.getsizeof(data2)))
    print ('Color transformation complete')
    del data
    gc.collect()

    return data2



def create_df(numbers_of_steps, chr_number, stepsize, width):
    #numbers_of_steps=int(math.floor((len(data[0])-width/200)/(stepsize/200)))
    step_number = range(0,numbers_of_steps)
    df = pd.DataFrame({'Step_Number':step_number})
    df['Start']=df['Step_Number']*stepsize
    df['End']=df['Start']+width
    return df

def apply_CNN_forward(data, chr_number, image_size, model, width, row):
    img = make_image_faster(data, chr_number, int(row['Start']), int(row['End']), width, 111)
    img = np.array(img)
    img = np.transpose(img, (1,0,2))
    img_resized = np.array(Image.fromarray(img).resize((image_size,111)))
    img_resized = np.transpose(img_resized, (1,0,2))
    img_resized = img_resized/255
    img_resized = np.expand_dims(img_resized, axis=0)
    return model.predict(img_resized)[0][0]

def apply_CNN_backward(data, chr_number, image_size, model, width, row):
    img = make_image_faster(data, chr_number, int(row['Start']), int(row['End']), width, 111)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = np.array(img)           
    img = np.transpose(img, (1,0,2))            
    img_resized = np.array(Image.fromarray(img).resize((image_size,111)))            
    img_resized = np.transpose(img_resized, (1,0,2))            
    img_resized=img_resized/255
    img_resized = np.expand_dims(img_resized, axis=0)
    return model.predict(img_resized)[0][0]


    
i=run_on_chr
chr_number = str(i)
print (chr_number)
print ("Load Data:")
data = load_chr_data(i, colorstate)
#print ("data " + str(sys.getsizeof(data)))
df = pd.DataFrame()
df = create_df(int(math.floor((len(data[0])-width/200)/(stepsize/200))), chr_number, stepsize, width)
#print ("df " + str(sys.getsizeof(df)))
print ('start predicting')
df['Forward']=df.apply(lambda row: apply_CNN_forward(data, chr_number, image_size, model, width, row), axis=1)
#print ("df " + str(sys.getsizeof(df)))
df['Backward']=df.apply(lambda row: apply_CNN_backward(data, chr_number, image_size, model, width, row), axis=1)
#print ("df " + str(sys.getsizeof(df)))
print (df.head())
df.to_csv(prjname + '/output/Predictions/Predictions_raw_chr' + chr_number + '.csv', sep='\t')

