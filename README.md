# dt_svm

Decision tree implementation and Support Vector Machines

In the first part, Decision tree with information gain and gain ratio models is implemented.
Binary tree is used. Threshold values were determined according to best entropy.

In the second part, SVM algorithm on a breast cancer dataset is utilized.
LIBSVM library was used. Min-max normalization is applied.


Base Environment

Create a virtual environment with Anaconda:

conda create -n 462assignment python=3.6 conda activate 462assignment

Load the requirements:

python3 -m pip install -r requirements.txt

Part1:

python3 dt_svm.py part1 step1
python3 dt_svm.py part1 step2

Part2:

python3 dt_svm.py part2 step1
python3 dt_svm.py part2 step2

#### Decision Tree ####

• Step1: DT is implemented with information gain and applied on Iris dataset.
• Step2: DT is implemented with gain ratio and applied on Iris dataset.

#### SVM ####

• Step1: SVM is applied to Breast Cancer Wisconsin dataset from UCI with 5 different C values for a fixed kernel.
• Step2:  SVM is applied to Breast Cancer Wisconsin dataset from UCI with different kernels for a fixed C value. 
