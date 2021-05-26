# sPLINK Client
The client component of **sPLINK** provides the basic functionalities inherited from the **HyFed** framework:  (1) joining the project, (2) obtaining the project info/parameters from 
the server, (3) selecting the dataset to contribute to the study, (4) keeping track the status/progress of the project, and (5) handling the communication 
with the server and compensator components. It implements required functions to open and pre-process a GWAS dataset as well as to compute the local model parameters such as 
sample size, non-missing sample size and allele count, contingency table for chi-square test, covariance matrix for linear regression test, and gradient vector, Hessian matrix, and log likelihood for logistic regression test.
