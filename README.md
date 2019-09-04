# Usage
* create a new folder, this represents a new experiment
* write a script which does the following:
    * clones this repository
    * checks out a specific revision (This represents a revision of the code which is used to run the experiment, since its in the repo, the revision identifies the exact code used to run the experiment)
    * invokes the python script with the necessary arguments, this ensures the arguments used for the experiment are saved.

_NOTE:_ At anytime, there would be only one experiment run inside a folder so the above steps are fine. If you need to execute a different revision of another script in this repo (e.g. vis.py) then
you would need to clone the repo again and then revert back to the specific revision to use the script. This is clumsy, but untill a better way is figured out, this would be our approach.
Code for deep learning models to predict molecular electronic properties. For the deep tensor neural network model, I built upon the code from Peter Bjørn Jørgensen (DTU) and would like to thank him for sharing his implementation. 

Code corresponding to the paper : Deep Learning Spectroscopy: Neural Networks for Molecular Excitation Spectra ([PDF](https://onlinelibrary.wiley.com/doi/full/10.1002/advs.201801367))


The code to reproduce the results in the paper are here : https://github.com/kunalghosh/Deep-Learning-Spectroscopy

To run the DTNN code use the following hyper parameters:

```shell
THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32,openmp=True python deep_tensor_refactored/train.py --learn_rate 0.00001 --clen 40 --batch_size 50 --num_neu_1 100 --num_neu_2 200 --model_name model_neu1_neu2_with_noise data.xyz spectra.npz
```

Running the CNN code is similar : 

```shell
THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32,openmp=True python /m/home/home0/00/ghoshk1/data/Desktop/Thesis/code/thesis_code/CNN_refactored/train.py --learn_rate 0.0001 --coulomb_dim 29 -c 22 -c 47 -c 42 --batch_size 90 --model_name orig_model coulomb.npz spectra.npz
```

The data that we trained on can be found here : https://zenodo.org/record/3386508

Pre-trained models can be found here : https://zenodo.org/record/3386531

To predict on new Coulomb matrices (we had a bug and trained on -ve of coulomb matrices so remember to do that when you predict on new CMs) with pertained CNN : 

```shell
THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32,openmp=True python CNN_refactored/predict.py --coulomb_dim 29 -c 22 -c 47 -c 42 --batch_size 90 --model_name orig_model $data_path/cm_10k_neg.txt $saved_model_path/results/model_epoch880.pkl.gz $saved_model_path/Y_vals.npz
```

To predict on new XYZ files using pertained DTNN :

```shell
OMP_NUM_THREADS=8 THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32,openmp=True python deep_tensor_refactored/predict.py --clen 40 --batch_size 50 --num_neu_1 100 --num_neu_2 200 --model_name model_neu1_neu2_with_noise $data_path/new.xyz $saved_model_path/results/model_epoch9998.pkl.gz $saved_model_path/Y_vals.npz
```
