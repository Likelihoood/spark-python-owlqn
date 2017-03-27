# spark-python-owlqn
L1 Logistic Regression optimized by OWLQN

I have search OWLQN in github, found that  there is no such implementation for LR.
SPARK has its own implementation for L1 by SGD, but it seems like that it is not suitable for L1.

So I try to implement OWLQN in python.

Input:
Sparse Vector define by following format:
(label, {index: value})
label can be 1, 0/-1
it use dict to describe vector, index is the feature index, value is the feature value


output:
weights
{index: weight}

it support intercept by setting intercept as True while create OWLQN instance.
intercept value will be the weights[feaNum]

LBFGS and backtrackline search will be executed in driver of spark, so it request driver memory can hold all feature weight and LBFGS's vector.

