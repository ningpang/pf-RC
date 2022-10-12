# pf-RC
## Paper:
[**Personalized Federated Relation Classification**]

## Dependencies
> python 3.6  
> torch > 1.2.0  
> numpy  
> sklearn
> 

# Run
All arguments are setted in the file argument.py. If you want run experiments on different datasets or with different non-IID data, please modify the hyperparameters in argument.py.

At present, we provide the data partition of SemEval, TACRED and TACREDV with `$\beta \in \{1, 5, 100\}$`.

## run pf-RC with heterogeneous local models: 
> python train.py --personal True 

## run pf-RC with homogeneous local models: 
> python train.py --personal False

