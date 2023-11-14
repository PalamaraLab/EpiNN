# 🧬 EpiNN

EpiNN was implemented and tested using CentOS Linux 7.

The training data is available in https://github.com/PalamaraLab/EpiNN_data

## 🔬 Get started

Run the following code in the "projects" directory.

You will need to have Conda installed. You can create and activate the Conda environment by running

```
conda create --name epinn
conda activate epinn
```

Then you can set up the conda environment by running the following commands (this will take a few minutes to complete)

```
pip install pandas
pip install csv
pip install numpy
pip install PIL
pip install gzip
pip install matplotlib
pip install sklearn
pip install tensorflow
pip install keras
pip install scipy
pip install gc
```


## 🧫 Prepare Data

- Download data and unzip:

Download the training data from https://github.com/PalamaraLab/EpiNN_data, then decompress the raw data in Data/Raw_data.tar.gz by typing 

```
tar -cvzf Raw_data.tar.gz
```

- Optimal parameters:

Chose a projectname, e.g.:
```
prjname=EpiNN
```

The following set of parameters can be used:
```
image_length=100000     (Image length in bp)
stepsize=10000          (Step size of the sliding window in bp)
colorstate=Color        (Color equals the full EpiNN color model)
image_size=1000         (Pixels per image)
cv=10                   (Number of cross-validation cycles)
```

- Training Data preparation:

run the following scripts to prepare the training data:

```
python 1_Preprocess_bed_files.py $prjname $image_length
python 2_Make_permanent_images.py $prjname $image_length $colorstate "Negative"
python 2_Make_permanent_images.py $prjname $image_length $colorstate "Positive"
```


## 🧪 Train EpiNN

- Training

To train EpiNN from scratch, run the following:
```
python -c "import tensorflow as tf; print(tf.version.GIT_VERSION, tf.version.VERSION)"

#Restrict Tensorflow to 1 CPU in node
export TF_NUM_INTEROP_THREADS="1"
export TF_NUM_INTRAOP_THREADS="4"

python 4_Run_CNN-DF_CV.py $prjname $cv 
```
where the second argument is the number of cross validation folds


## 🧬 Apply EpiNN

- Apply to whole genome using a sliding window
```
for $chr in $(seq 1 22); do
  python 6_SlidingWindow.py $prjname $image_length $stepsize $colorstate $image_size $chr
done
python 6_SlidingWindow.py $prjname $image_length $stepsize $colorstate $image_size X
python 6_SlidingWindow.py $prjname $image_length $stepsize $colorstate $image_size Y
```
where the second argument is the image length in bp, the third the stepsize in bp, the fifth a hyperparemeter ("image_size") related to the number of pixels per image.

- Post-processing:
```
for $chr in $(seq 1 22); do
  python 7_Postprocessing.py $prjname $image_length $stepsize $chr
done
python 7_Postprocessing.py $prjname $image_length $stepsize X
python 7_Postprocessing.py $prjname $image_length $stepsize Y
bash 7b_Postprocessing_ave.sge.sh $prjname
```

This should result in a .bed file with predicted IS relevance for each locus in the genome.
