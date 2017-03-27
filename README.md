# spark-python-owlqn
L1 Logistic Regression optimized by OWLQN

I have searched OWLQN in github, found that  there is no such implementation for L1 LR optimized by OWLQN in SPARK.
SPARK has its own implementation for L1 by SGD, but it seems like that it is not suitable for L1 especially for speacial case such as CTR, CVR model by using large scale sparse feature.
So I try to implement OWLQN in python.

Input:<br>
Sparse Vector define by following format:<br>
(label, {index: value})<br>
label can be 1, 0/-1<br>
it use dict to describe vector, index is the feature index, value is the feature value<br>

output:<br>
weights<br>
{index: weight}<br>

it support intercept by setting intercept as True while create OWLQN instance.<br>

intercept value will be the weights[feaNum]<br>

LBFGS and backtrackline search will be executed in driver of spark, so it request driver memory can hold all feature weight and LBFGS's vector.<br>

RUN Test:<br>

```python
pyspark

>>import test
>>test.testTrain(sc)

call calLossAndGradient() in 0.776247s
============iterator : 0 ==========
call virtualGradient() in 0.000022s
call LBFGS() in 0.000004s
call fixDirection() in 0.000005s
call calLossAndGradient() in 0.152505s
call backTrackingLineSearch() in 0.161142s
please wait for more than 5 times iterator
call shift() in 0.000023s
============iterator : 0 end ==========

============iterator : 1 ==========
call virtualGradient() in 0.000011s
call LBFGS() in 0.000076s
call fixDirection() in 0.000004s
call calLossAndGradient() in 0.125030s
call backTrackingLineSearch() in 0.131714s
please wait for more than 5 times iterator
call shift() in 0.000024s
============iterator : 1 end ==========

...


============iterator : 24 ==========
call virtualGradient() in 0.000011s
call LBFGS() in 0.000142s
call fixDirection() in 0.000003s
call calLossAndGradient() in 0.111331s
call backTrackingLineSearch() in 0.116443s
iterator: 24, oldLoss: 0.210383096226, newLoss: 0.209059351947, reduceLoss: 0.000132374427879 , reduceRatio:0.000633190654453, intercept: -5.70113778841
call train() in 4.395765s


0 -0.511179076907
1 11.6898770259
2 0.0
3 0.0
4 0.0
5 0.0
6 0.0
7 0.0
8 0.0
9 0.0
10 -5.70113778841
1 0.997499445879
1 0.997499445879
-1 0.0033310277918
-1 0.00200057738711
-1 0.00200057738711
-1 0.0033310277918
-1 0.00200057738711
-1 0.0033310277918
1 0.997499445879
1 0.997499445879
-1 0.00200057738711
-1 0.00200057738711
```
