#!/usr/bin/env python
# coding: utf-8

# # Poisoning Attacks against Machine Learning models
# 
# In this tutorial we will experiment with **adversarial poisoning attacks** 
#  against a Support Vector Machine (SVM) with Radial Basis Function (RBF) kernel.
# 
# Poisoning attacks are performed at *train time* by injecting *carefully crafted 
#  samples* that alter the classifier decision function so that its accuracy decreases.
# 
# As in the previous tutorials, we will first create and train the classifier, 
#  evaluating its performance in the standard scenario, *i.e. not under attack*.
#  The poisoning attack will also need a *validation set* to verify the classifier
#  performance during the attack, so we split the training set furtherly in two.
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
# https://colab.research.google.com/github/pralab/secml/blob/HEAD/tutorials/05-Poisoning.ipynb)

# In[1]:


import sklearn
import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

mms = MinMaxScaler()
pca = PCA(n_components=8)

mnist = fetch_openml('mnist_784')
X_orig, y_orig = mnist["data"], mnist["target"]

X_orig = mms.fit_transform(np.array(X_orig).astype(np.float32))
y_orig = np.array(y_orig).astype(np.uint8)
target_digit1 = 3
target_digit2 = 8

target_digit1_xdata = X_orig[y_orig == target_digit1]
target_digit2_xdata = X_orig[y_orig == target_digit2]
target_digit1_ydata = y_orig[y_orig == target_digit1]
target_digit2_ydata = y_orig[y_orig == target_digit2]
X = np.concatenate((target_digit1_xdata, target_digit2_xdata), axis=0)
y = np.concatenate((target_digit1_ydata, target_digit2_ydata), axis=0)
y = np.where(y == target_digit1, 0, 1)
X = pca.fit_transform(X)
X = mms.fit_transform(X)


# In[2]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

pca.fit(X_train)

print(X_train.shape, X_test.shape, X_val.shape)
print(y_train.shape, y_test.shape, y_val.shape)
# In[2]:


# In[3]:


from secml.data import CDataset
from secml.array.c_dense import CDense

tr = CDataset(X_train, y_train)
ts = CDataset(X_test, y_test)
val = CDataset(X_val, y_val)


# In[4]:


random_state = 999

# n_features = 2  # Number of features
# n_samples = 300  # Number of samples
# centers = [[-1, -1], [+1, +1]]  # Centers of the clusters
# cluster_std = 0.9  # Standard deviation of the clusters
#
# from secml.data.loader import CDLRandomBlobs
# dataset = CDLRandomBlobs(n_features=n_features,
#                          centers=centers,
#                          cluster_std=cluster_std,
#                          n_samples=n_samples,
#                          random_state=random_state).load()
#
# n_tr = 100  # Number of training set samples
# n_val = 100  # Number of validation set samples
# n_ts = 100  # Number of test set samples
#
# # Split in training, validation and test
# from secml.data.splitter import CTrainTestSplit
# splitter = CTrainTestSplit(
#     train_size=n_tr + n_val, test_size=n_ts, random_state=random_state)
# tr_val, ts = splitter.split(dataset)
# splitter = CTrainTestSplit(
#     train_size=n_tr, test_size=n_val, random_state=random_state)
# tr, val = splitter.split(dataset)
#
# # Normalize the data
# from secml.ml.features import CNormalizerMinMax
# nmz = CNormalizerMinMax()
# tr.X = nmz.fit_transform(tr.X)
# val.X = nmz.transform(val.X)
# ts.X = nmz.transform(ts.X)

# Metric to use for training and performance evaluation
from secml.ml.peval.metrics import CMetricAccuracy
metric = CMetricAccuracy()

# Creation of the multiclass classifier
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelRBF
clf = CClassifierSVM(kernel=CKernelRBF(gamma=0.1), C=1)

# We can now fit the classifier
clf.fit(tr.X, tr.Y)
print("Training of classifier complete!")

# Compute predictions on a test set
y_pred = clf.predict(ts.X)


# In[5]:


# Compute predictions on a test set
y_pred_np = y_pred.tondarray()
y_test_np = ts.Y.tondarray()
from sklearn import metrics
accuracy_np = metrics.accuracy_score(y_test_np, y_pred_np)
precision_np = metrics.precision_score(y_test_np, y_pred_np)
recall_np = metrics.recall_score(y_test_np, y_pred_np)
cm_np = metrics.confusion_matrix(y_test_np, y_pred_np)
# hinge = metrics.hinge_loss(y_test, svm_clf.decision_function(X_test))
print('Accuracy: %.4f' % (accuracy_np))
print('Precision: %.4f' % (precision_np))
print('Recall: %.4f' % (recall_np))
# print('Hinge loss: %.4f' % hinge)
print(cm_np)


# ## Generation of Poisoning Samples
# 
# We are going to generate an adversarial example against the SVM classifier
#  using the **gradient-based** algorithm for generating poisoning attacks 
#  proposed in: 
#  
#   > [[biggio12-icml]](https://arxiv.org/abs/1206.6389)
#   > Biggio, B., Nelson, B. and Laskov, P., 2012. Poisoning attacks against 
#   > support vector machines. In ICML 2012.
# 
#   > [[biggio15-icml]](https://arxiv.org/abs/1804.07933)
#   > Xiao, H., Biggio, B., Brown, G., Fumera, G., Eckert, C. and Roli, F., 2015. 
#   > Is feature selection secure against training data poisoning?. In ICML 2015.
# 
#   > [[demontis19-usenix]](
#   > https://www.usenix.org/conference/usenixsecurity19/presentation/demontis)
#   > Demontis, A., Melis, M., Pintor, M., Jagielski, M., Biggio, B., Oprea, A., 
#   > Nita-Rotaru, C. and Roli, F., 2019. Why Do Adversarial Attacks Transfer? 
#   > Explaining Transferability of Evasion and Poisoning Attacks. In 28th Usenix 
#   > Security Symposium, Santa Clara, California, USA.
# 
# To compute a poisoning point, a bi-level optimization problem has to be solved, namely:
# 
# $$
# \begin{aligned}
# \max_{x_c}& A(D_{val}, \mathbf{w}^\ast) = \sum_{j=1}^m \ell(y_j, \mathbf{x_\mathit{j}}, \mathbf{w}^\ast)\\
# &s.t. \mathbf{w}^\ast \in \underset{\mathbf{w}}{\operatorname{arg min}} \textit{L} (D_{tr} \cup (\mathbf{x}_c, y_c), \mathbf{w})
# \end{aligned}
# $$
# 
# Where $\mathbf{x_c}$ is the poisoning point, $A$ is the attacker objective function, $L$ is the classifier training
# function. Moreover, $D_{tr}$ is the training dataset and $D_{val}$ is the validation dataset.
# The former problem, along with the poisoning point $\mathbf{x}_c$ is used to train the classifier on the poisoned data,
# while the latter is used to evaluate the performance on the untainted data.
# 
# The former equation depends on the classifier weights, which in turns, depends on the poisoning point.
# 
# This attack is implemented in SecML by different subclasses of the `CAttackPoisoning`.
#  For the purpose of attacking a SVM classifier we use the `CAttackPoisoningSVM` 
#  class.
# 
# As done for the [evasion attacks](03-Evasion.ipynb), let's specify the 
#  parameters first. We set the bounds of the attack space to the known feature
#  space given by validation dataset. Lastly, we chose the solver parameters for this specific optimization problem.
# 
# Let's start visualizing the objective function considering a single poisoning point.

# In[6]:


lb, ub = val.X.min(), val.X.max()  # Bounds of the attack space. Can be set to `None` for unbounded

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.05,
    'eta_min': 0.05,
    'eta_max': None,
    'max_iter': 100,
    'eps': 1e-6
}

from poison import CAttackPoisoningSVM
pois_attack = CAttackPoisoningSVM(classifier=clf,
                                  training_data=tr,
                                  val=val,
                                  lb=lb, ub=ub,
                                  solver_params=solver_params,
                                  random_seed=random_state)

# chose and set the initial poisoning sample features and label
xc = tr[0,:].X
yc = tr[0,:].Y
pois_attack.x0 = xc
pois_attack.xc = xc
pois_attack.yc = yc

print("Initial poisoning sample features: {:}".format(xc.ravel()))
print("Initial poisoning sample label: {:}".format(yc.item()))




# Now, we set the desired number of adversarial points to generate, 20 in this example.

# In[12]:


# 10 points = 2 minutes
n_poisoning_points = 1000  # Number of poisoning points to generate
pois_attack.n_points = n_poisoning_points

# Run the poisoning attack
print("Attack started...")
pois_y_pred, pois_scores, pois_ds, f_opt = pois_attack.run(ts.X, ts.Y)
print("Attack complete!")

# Evaluate the accuracy of the original classifier
acc = metric.performance_score(y_true=ts.Y, y_pred=y_pred)
# Evaluate the accuracy after the poisoning attack
pois_acc = metric.performance_score(y_true=ts.Y, y_pred=pois_y_pred)

print("Original accuracy on test set: {:.2%}".format(acc))
print("Accuracy after attack on test set: {:.2%}".format(pois_acc))


# We can see that the classifiers has been successfully attacked. To increase 
# the attack power, more poisoning points can be crafted, at the expense of 
#  a much slower optimization process.
# 
# Let's now visualize the attack on a 2D plane. We need to train a copy of the 
# original classifier on the join between the training set and the poisoning points.

# In[9]:


# Training of the poisoned classifier
pois_clf = clf.deepcopy()
pois_tr = tr.append(pois_ds)  # Join the training set with the poisoning points
pois_clf.fit(pois_tr.X, pois_tr.Y)

# Define common bounds for the subplots
min_limit = min(pois_tr.X.min(), ts.X.min())
max_limit = max(pois_tr.X.max(), ts.X.max())
grid_limits = [[min_limit, max_limit], [min_limit, max_limit]]

fig = CFigure(10, 10)

fig.subplot(2, 2, 1)
fig.sp.title("Original classifier (training set)")
fig.sp.plot_decision_regions(
    clf, n_grid_points=200, grid_limits=grid_limits)
fig.sp.plot_ds(tr, markersize=5)
fig.sp.grid(grid_on=False)

fig.subplot(2, 2, 2)
fig.sp.title("Poisoned classifier (training set + poisoning points)")
fig.sp.plot_decision_regions(
    pois_clf, n_grid_points=200, grid_limits=grid_limits)
fig.sp.plot_ds(tr, markersize=5)
fig.sp.plot_ds(pois_ds, markers=['*', '*'], markersize=12)
fig.sp.grid(grid_on=False)

fig.subplot(2, 2, 3)
fig.sp.title("Original classifier (test set)")
fig.sp.plot_decision_regions(
    clf, n_grid_points=200, grid_limits=grid_limits)
fig.sp.plot_ds(ts, markersize=5)
fig.sp.text(0.05, -0.25, "Accuracy on test set: {:.2%}".format(acc), 
            bbox=dict(facecolor='white'))
fig.sp.grid(grid_on=False)

fig.subplot(2, 2, 4)
fig.sp.title("Poisoned classifier (test set)")
fig.sp.plot_decision_regions(
    pois_clf, n_grid_points=200, grid_limits=grid_limits)
fig.sp.plot_ds(ts, markersize=5)
fig.sp.text(0.05, -0.25, "Accuracy on test set: {:.2%}".format(pois_acc), 
            bbox=dict(facecolor='white'))
fig.sp.grid(grid_on=False)

fig.show()


# We can see how the SVM classifier decision functions *changes* after injecting
#  the adversarial poisoning points (blue and red stars).
#  
# For more details about poisoning adversarial attacks please refer to:
# 
#   > [[biggio18-pr]](https://arxiv.org/abs/1712.03141)
#   > Biggio, B. and Roli, F., 2018. Wild patterns: Ten years after the rise of 
#   > adversarial machine learning. In Pattern Recognition.
