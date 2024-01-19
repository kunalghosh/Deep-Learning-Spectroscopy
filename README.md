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

# Software versions used:
* CUDA - 8.0.61 (did not use CuDNN)
* Python - 2.7
* Theano - 0.9.0
* PyGPU - 0.6.4
* Lasagne - 0.1
* NumPy - 1.12.1
* SciPy - 0.19.0

To run the DTNN code use the following hyper parameters:

```shell
THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32,openmp=True python deep_tensor_refactored/train.py --learn_rate 0.00001 --clen 40 --batch_size 50 --num_neu_1 100 --num_neu_2 200 --model_name model_neu1_neu2_with_noise data.xyz spectra.npz
```

Running the CNN code is similar : 

```shell
THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32,openmp=True python /m/home/home0/00/ghoshk1/data/Desktop/Thesis/code/thesis_code/CNN_refactored/train.py --learn_rate 0.0001 --coulomb_dim 29 -c 22 -c 47 -c 42 --batch_size 90 --model_name orig_model coulomb.npz spectra.npz
```

The data that we trained on can be found here : https://zenodo.org/record/3386508

:exclamation: **NOTE** : After publishing the paper we noticed that about twelve molecules in the dataset had less than 16 energy levels.
*To be consistent with the text in the paper, please remove these molecules when running the experiments.*
We anticipate the results to not change significantly, since a very small portion (12 molecules out of 132k) of the dataset is affected. The [following script](https://colab.research.google.com/drive/1uu8vEDEYAzKRVklVmf6iuqg7PoHnW5I0) was used to identify the molecules.

Index in the dataset | Molecule IUPAC |
---------------------|----------------|
| 0 | CH4 |
 | 1 | C2H6 |
 | 2 | C3H4 |
 | 3 | C2OH4 |
 | 4 | C3H8 |
 | 5 | C2OH6 |
 | 6 | COCH6 |
 | 7 | C3H6 |
 | 8 | C2OH4 |
 | 10 | C4H2 |
 | 11 | C4H6 |
 | 12 | C4H6 |
 
 :exclamation: **NOTE** :Also the test set and validation set combined is about 13000 molecules for the 134k dataset, the test set had half the number of molecules i.e. 6627. We thank [Kanishka Singh](https://www.heibrids.berlin/people/doctoral-researchers/kanishka-singh/) for pointing out this typo.

Pre-trained models can be found here : https://zenodo.org/record/3386531

To predict on new Coulomb matrices (we had a bug and trained on -ve of coulomb matrices so remember to do that when you predict on new CMs) with pertained CNN : 

```shell
THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32,openmp=True python CNN_refactored/predict.py --coulomb_dim 29 -c 22 -c 47 -c 42 --batch_size 90 --model_name orig_model $data_path/cm_10k_neg.txt $saved_model_path/results/model_epoch880.pkl.gz $saved_model_path/Y_vals.npz
```

To predict on new XYZ files using pertained DTNN :

```shell
OMP_NUM_THREADS=8 THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32,openmp=True python deep_tensor_refactored/predict.py --clen 40 --batch_size 50 --num_neu_1 100 --num_neu_2 200 --model_name model_neu1_neu2_with_noise $data_path/new.xyz $saved_model_path/results/model_epoch9998.pkl.gz $saved_model_path/Y_vals.npz
```

## Idea behind the Root square error (RSE) metric:

With the RSE we wanted to present a scaled metric. Which gives some measure of how much is the error in the predicted spectra compared to the original target spectra.

The spectra we used are computed in the range : [-30 eV, 0 eV] and are discretized into 300 points (```len(prediction)``` in the code)

Given a predicted and target spectra we compute the RSE as follows:
```python
def relative_difference(prediction, target):
    dE = 30/len(prediction) #how many eV's one dE is
    numerator = np.sum(dE*np.power((target-prediction),2))
    denominator = np.sum(dE*target)
    
    return np.sqrt(numerator)/denominator
```

# Cite Us
```
@article{ghosh_dlspectroscopy_2019,
author = {Ghosh, Kunal and Stuke, Annika and Todorović, Milica and Jørgensen, Peter Bjørn and Schmidt, Mikkel N. and Vehtari, Aki and Rinke, Patrick},
title = {Deep Learning Spectroscopy: Neural Networks for Molecular Excitation Spectra},
journal = {Advanced Science},
volume = {6},
number = {9},
pages = {1801367},
keywords = {artificial intelligence, DFT calculations, excitation spectra, neural networks, organic molecules},
doi = {https://doi.org/10.1002/advs.201801367},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/advs.201801367},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/advs.201801367},
abstract = {Abstract Deep learning methods for the prediction of molecular excitation spectra are presented. For the example of the electronic density of states of 132k organic molecules, three different neural network architectures: multilayer perceptron (MLP), convolutional neural network (CNN), and deep tensor neural network (DTNN) are trained and assessed. The inputs for the neural networks are the coordinates and charges of the constituent atoms of each molecule. Already, the MLP is able to learn spectra, but the root mean square error (RMSE) is still as high as 0.3 eV. The learning quality improves significantly for the CNN (RMSE = 0.23 eV) and reaches its best performance for the DTNN (RMSE = 0.19 eV). Both CNN and DTNN capture even small nuances in the spectral shape. In a showcase application of this method, the structures of 10k previously unseen organic molecules are scanned and instant spectra predictions are obtained to identify molecules for potential applications.},
year = {2019}
}
```
