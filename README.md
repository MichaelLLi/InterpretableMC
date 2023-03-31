# Interpretable Matrix Completion

This is the code repository for the paper "Interpretable Matrix Completion: A Discrete Optimization Approach". 

## Synthetic Experiments

The code for the implementation of our algorithm can be found in the file IMC.jl. The function *MatrixOptIntFull* implements CutPlanes in the paper while *MatrixOptInt* implement OptComplete. 

It has been tested on Julia V1.4, and we provide a demonstration code on how one can reproduce our synthetic experiments. 


### Real data experiments

To replicate the real data experiments, please download the Netflix Prize Data from e.g. https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data, and 
the corresponding TMDB data using the API endpoint https://www.themoviedb.org/documentation/api?language=en-US.

Then the algorithm can be ran according to the demonstration code in IMC.jl. 

If there are any questions, please reach out to Michael Lingzhi Li at mili@hbs.edu. 