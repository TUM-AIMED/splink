# sPLINK Server
The server component of sPLINK provides the basic functionalities inherited from the HyFed framework: (1) backend part to handle the WebApp requests such as 
creating the project and tokens, viewing the results, etc; (2) coordinating the training process;
(3) measuring runtime and network bandwidth usage of the algorithms. It implements the aggregation logic to compute chi-square,
 linear/logistic regression related global parameters such as the global non-missing sample count, allele count, contingency table,
covariance matrix, Hessian matrix, Gradient vector, and standard error values. It also computes the final results such as beta values, t-stat values, and p-values,
and stores them into the file.
