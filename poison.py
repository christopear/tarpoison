import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from secml.array import CArray
from secml.data import CDataset
from secml.ml.classifiers.loss import CLoss
from secml.ml.peval.metrics import CMetric
from secml.optim.constraints import CConstraint
from secml.optim.function import CFunction
from secml.optim.optimizers import COptimizer


class CAttackPoisoningSVM:
    """Poisoning attacks against Support Vector Machines (SVMs).

    This is an implementation of the attack in https://arxiv.org/pdf/1206.6389:

     - B. Biggio, B. Nelson, and P. Laskov. Poisoning attacks against
       support vector machines. In J. Langford and J. Pineau, editors,
       29th Int'l Conf. on Machine Learning, pages 1807-1814. Omnipress, 2012.

    where the gradient is computed as described in Eq. (10) in
    https://www.usenix.org/conference/usenixsecurity19/presentation/demontis:

     - A. Demontis, M. Melis, M. Pintor, M. Jagielski, B. Biggio, A. Oprea,
       C. Nita-Rotaru, and F. Roli. Why do adversarial attacks transfer?
       Explaining transferability of evasion and poisoning attacks.
       In 28th USENIX Security Symposium. USENIX Association, 2019.

    For more details on poisoning attacks, see also:

     - https://arxiv.org/abs/1804.00308, IEEE Symp. SP 2018
     - https://arxiv.org/abs/1712.03141, Patt. Rec. 2018
     - https://arxiv.org/abs/1708.08689, AISec 2017
     - https://arxiv.org/abs/1804.07933, ICML 2015

    Parameters
    ----------
    classifier : CClassifierSVM
        Target SVM, trained in the dual (i.e., with kernel not set to None).
    training_data : CDataset
        Dataset on which the the classifier has been trained on.
    val : CDataset
        Validation set.
    distance : {'l1' or 'l2'}, optional
        Norm to use for computing the distance of the adversarial example
        from the original sample. Default 'l2'.
    dmax : scalar, optional
        Maximum value of the perturbation. Default 1.
    lb, ub : int or CArray, optional
        Lower/Upper bounds. If int, the same bound will be applied to all
        the features. If CArray, a different bound can be specified for each
        feature. Default `lb = 0`, `ub = 1`.
    y_target : int or None, optional
        If None an error-generic attack will be performed, else a
        error-specific attack to have the samples misclassified as
        belonging to the `y_target` class.
    solver_type : str or None, optional
        Identifier of the solver to be used. Default 'pgd-ls'.
    solver_params : dict or None, optional
        Parameters for the solver. Default None, meaning that default
        parameters will be used.
    init_type : {'random', 'loss_based'}, optional
        Strategy used to chose the initial random samples. Default 'random'.
    random_seed : int or None, optional
        If int, random_state is the seed used by the random number generator.
        If None, no fixed seed will be set.

    """
    __class_type = 'p-svm'

    def __init__(self, classifier,
                 training_data,
                 val,
                 distance='l1',
                 dmax=0,
                 lb=0,
                 ub=1,
                 y_target=None,
                 solver_type='pgd-ls',
                 solver_params=None,
                 init_type='random',
                 random_seed=None):

        self.classifier = classifier

        # These are internal parameters populated by _run,
        # for the *last* attack point:
        self._x_opt = None  # the final/optimal attack point
        self._f_opt = None  # the objective value at the optimum
        self._x_seq = None  # the path of points through the optimization
        self._f_seq = None  # the objective values along the optimization path

        # INTERNAL
        # init attributes to None (re-defined through setters below)
        self._solver = None

        # now we populate solver parameters (via dedicated setters)
        self.dmax = dmax
        self.lb = lb
        self.ub = ub
        self.distance = distance
        self.solver_type = solver_type
        self.solver_params = solver_params

        # fixme: validation loss should be optional and passed from outside
        if classifier.class_type == 'svm':
            loss_name = 'hinge'
        elif classifier.class_type == 'logistic':
            loss_name = 'log'
        elif classifier.class_type == 'ridge':
            loss_name = 'square'
        else:
            raise NotImplementedError("We cannot poisoning that classifier")

        self._attacker_loss = CLoss.create(
            loss_name)

        self._init_loss = self._attacker_loss

        self.y_target = y_target

        # hashing xc to avoid re-training clf when xc does not change
        self._xc_hash = None

        self._x0 = None  # set the initial poisoning sample feature
        self._xc = None  # set of poisoning points along with their labels yc
        self._yc = None
        self._idx = None  # index of the current point to be optimized
        self._training_data = None  # training set used to learn classifier
        self.n_points = None  # FIXME: INIT PARAM?

        # READ/WRITE
        self.val = val  # this is for validation set
        self.training_data = training_data
        self.random_seed = random_seed

        self.init_type = init_type

        self.eta = solver_params['eta']

        # this is used to speed up some poisoning algorithms by re-using
        # the solution obtained at a previous step of the optimization
        self._warm_start = None

        # check if SVM has been trained in the dual
        if self.classifier.kernel is None:
            raise ValueError(
                "Please retrain the SVM in the dual (kernel != None).")

        # indices of support vectors (at previous iteration)
        # used to check if warm_start can be used in the iterative solver
        self._sv_idx = None

    ###########################################################################
    #                           PRIVATE METHODS
    ###########################################################################

    def _constraint_creation(self):

        # only feature increments or decrements are allowed
        lb = self._x0 if self.lb == 'x0' else self.lb
        ub = self._x0 if self.ub == 'x0' else self.ub
        bounds = CConstraint.create('box', lb=lb, ub=ub)

        constr = CConstraint.create(self.distance, center=0, radius=1e12)

        return bounds, constr
    def _update_poisoned_clf(self, clf=None, tr=None,
                             train_normalizer=False):
        """
        Trains classifier on D (original training data) plus {x,y} (new point).

        Parameters
        ----------
        x: feature vector of new training point
        y: true label of new training point

        Returns
        -------
        clf: trained classifier on D and {x,y}

        """

        #  xc hashing is only valid if clf and tr do not change
        #  (when calling update_poisoned_clf() without parameters)
        xc_hash_is_valid = False
        if clf is None and tr is None:
            xc_hash_is_valid = True

        if clf is None:
            clf = self.classifier

        if tr is None:
            tr = self.training_data

        tr = tr.append(CDataset(self._xc, self._yc))

        xc_hash = self._xc.sha1()

        if self._xc_hash is None or self._xc_hash != xc_hash:
            # xc set has changed, retrain clf
            # hash is stored only if update_poisoned_clf() is called w/out pars
            self._xc_hash = xc_hash if xc_hash_is_valid else None
            self._poisoned_clf = clf.deepcopy()

            # we assume that normalizer is not changing w.r.t xc!
            # so we avoid re-training the normalizer on dataset including xc

            if self.classifier.preprocess is not None:
                self._poisoned_clf.retrain_normalizer = train_normalizer

            self._poisoned_clf.fit(tr.X, tr.Y)

        return self._poisoned_clf, tr


    def objective_function(self, xc, acc=False):
        """
        Parameters
        ----------
        xc: poisoning point

        Returns
        -------
        f_obj: values of objective function (average hinge loss) at x
        """

        # index of poisoning point within xc.
        # This will be replaced by the input parameter xc
        if self._idx is None:
            idx = 0
        else:
            idx = self._idx

        xc = CArray(xc).atleast_2d()

        n_samples = xc.shape[0]
        if n_samples > 1:
            raise TypeError("xc is not a single sample!")

        self._xc[idx, :] = xc
        clf, tr = self._update_poisoned_clf()

        y_pred, score = clf.predict(self.val.X, return_decision_function=True)

        # targeted attacks
        y_ts = CArray(self.y_target).repeat(score.shape[0]) \
            if self.y_target is not None else self.val.Y

        # TODO: binary loss check
        if self._attacker_loss.class_type != 'softmax':
            score = CArray(score[:, 1].ravel())

        if acc is True:
            error = CArray(y_ts != y_pred).ravel()  # compute test error
        else:
            error = self._attacker_loss.loss(y_ts, score)
        obj = error.mean()

        return obj

    def objective_function_gradient(self, xc, normalization=True):
        """
        Compute the loss derivative wrt the attack sample xc

        The derivative is decomposed as:

        dl / x = sum^n_c=1 ( dl / df_c * df_c / x )
        """

        xc = xc.atleast_2d()
        n_samples = xc.shape[0]

        if n_samples > 1:
            raise TypeError("x is not a single sample!")

        # index of poisoning point within xc.
        # This will be replaced by the input parameter xc
        if self._idx is None:
            idx = 0
        else:
            idx = self._idx

        self._xc[idx, :] = xc
        clf, tr = self._update_poisoned_clf()

        # computing gradient of loss(y, f(x)) w.r.t. f
        _, score = clf.predict(self.val.X, return_decision_function=True)

        y_ts = CArray(self.y_target).repeat(score.shape[0]) \
            if self.y_target is not None else self.val.Y

        grad = CArray.zeros((xc.size,))

        if clf.n_classes <= 2:
            loss_grad = self._attacker_loss.dloss(
                y_ts, CArray(score[:, 1]).ravel())
            grad = self._gradient_fk_xc(
                self._xc[idx, :], self._yc[idx], clf, loss_grad, tr)
        else:
            # compute the gradient as a sum of the gradient for each class
            for c in range(clf.n_classes):
                loss_grad = self._attacker_loss.dloss(y_ts, score, c=c)

                grad += self._gradient_fk_xc(self._xc[idx, :], self._yc[idx],
                                             clf, loss_grad, tr, c)

        if normalization:
            norm = grad.norm()
            return grad / norm if norm > 0 else grad
        else:
            return grad

    def _init_solver(self):
        """Overrides _init_solver to additionally reset the SV indices."""

        if self.classifier is None:
            raise ValueError('Solver not set properly!')

        # map attributes to fun, constr, box
        fun = CFunction(fun=self.objective_function,
                        gradient=self.objective_function_gradient,
                        n_dim=self.classifier.n_features)

        bounds, constr = self._constraint_creation()

        self._solver = COptimizer.create(
            self.solver_type,
            fun=fun, constr=constr,
            bounds=bounds,
            **self.solver_params)

        self._solver.verbose = 0
        self._warm_start = None

        # reset stored indices of SVs
        self._sv_idx = None

    ###########################################################################
    #                  OBJECTIVE FUNCTION & GRAD COMPUTATION
    ###########################################################################

    def _alpha_c(self, clf):
        """
        Returns alpha value of xc, assuming xc to be appended
        as the last point in tr
        """

        # index of poisoning point within xc.
        # This will be replaced by the input parameter xc
        if self._idx is None:
            idx = 0
        else:
            idx = self._idx

        # index of the current poisoning point in the set self._xc
        # as this set is appended to the training set, idx is shifted
        idx += self.training_data.num_samples

        # k is the index of sv_idx corresponding to the training idx of xc
        k = clf.sv_idx.find(clf.sv_idx == idx)
        if len(k) == 1:  # if not empty
            alpha_c = clf.alpha[k].todense().ravel()
            return alpha_c
        return 0

    ###########################################################################
    #                            GRAD COMPUTATION
    ###########################################################################

    def _Kd_xc(self, clf, alpha_c, xc, xk):
        """
        Derivative of the kernel w.r.t. a training sample xc

        Parameters
        ----------
        xk : CArray
            features of a validation set
        xc:  CArray
            features of the training point w.r.t. the derivative has to be
            computed
        alpha_c:  integer
            alpha value of the of the training point w.r.t. the derivative has
            to be
            computed
        """
        # handle normalizer, if present
        p = clf.kernel.preprocess
        # xc = xc if p is None else p.forward(xc, caching=False)
        xk = xk if p is None else p.forward(xk, caching=False)

        rv = clf.kernel.rv
        clf.kernel.rv = xk
        dKkc = alpha_c * clf.kernel.gradient(xc)
        clf.kernel.rv = rv
        return dKkc.T  # d * k

    def _gradient_fk_xc(self, xc, yc, clf, loss_grad, tr, k=None):
        """
        Derivative of the classifier's discriminant function f(xk)
        computed on a set of points xk w.r.t. a single poisoning point xc
        """

        xc0 = xc.deepcopy()
        d = xc.size
        grad = CArray.zeros(shape=(d,))  # gradient in input space
        alpha_c = self._alpha_c(clf)

        if abs(alpha_c) == 0:  # < svm.C:  # this include alpha_c == 0
            # self.logger.debug("Warning: xc is not an error vector.")
            return grad

        # take only validation points with non-null loss
        xk = self._val.X[abs(loss_grad) > 0, :].atleast_2d()
        grad_loss_fk = CArray(loss_grad[abs(loss_grad) > 0]).T

        # gt is the gradient in feature space
        # this gradient component is the only one if margin SV set is empty
        # gt is the derivative of the loss computed on a validation
        # set w.r.t. xc
        Kd_xc = self._Kd_xc(clf, alpha_c, xc, xk)
        assert (clf.kernel.rv.shape[0] == clf.alpha.shape[1])

        gt = Kd_xc.dot(grad_loss_fk).ravel()  # gradient of the loss w.r.t. xc

        xs, sv_idx = clf._sv_margin()  # these points are already normalized

        if xs is None:
            self.logger.debug("Warning: xs is empty "
                              "(all points are error vectors).")
            return gt if clf.kernel.preprocess is None else \
                clf.kernel.preprocess.gradient(xc0, w=gt)

        s = xs.shape[0]

        # derivative of the loss computed on a validation set w.r.t. the
        # classifier params
        fd_params = clf.grad_f_params(xk)
        grad_loss_params = fd_params.dot(grad_loss_fk)

        H = clf.hessian_tr_params()
        H += 1e-9 * CArray.eye(s + 1)

        # handle normalizer, if present
        # xc = xc if clf.preprocess is None else clf.kernel.transform(xc)
        G = CArray.zeros(shape=(gt.size, s + 1))
        rv = clf.kernel.rv
        clf.kernel.rv = xs
        G[:, :s] = clf.kernel.gradient(xc).T
        clf.kernel.rv = rv
        G *= alpha_c

        # warm start is disabled if the set of SVs changes!
        # if self._sv_idx is None or self._sv_idx.size != sv_idx.size or \
        #         (self._sv_idx != sv_idx).any():
        #     self._warm_start = None
        # self._sv_idx = sv_idx  # store SV indices for the next iteration
        #
        # # iterative solver
        # v = - self._compute_grad_solve_iterative(
        #     G, H, grad_loss_params, tol=1e-3)

        # solve with standard linear solver
        # v = - self._compute_grad_solve(G, H, grad_loss_params, sym_pos=False)

        # solve using inverse/pseudo-inverse of H
        # v = - self._compute_grad_inv(G, H, grad_loss_params)
        v = self._compute_grad_inv(G, H, grad_loss_params)

        gt += v

        # propagating gradient back to input space
        if clf.kernel.preprocess is not None:
            return clf.kernel.preprocess.gradient(xc0, w=gt)

        return gt

    def _compute_grad_inv(self, G, H, grad_loss_params):

        from scipy import linalg
        det = linalg.det(H.tondarray())
        if abs(det) < 1e-6:
            H_inv = CArray(linalg.pinv(H.tondarray()))
        else:
            H_inv = CArray(linalg.inv(H.tondarray()))
        grad_mat = - CArray(G.dot(H_inv))  # d * (d + 1)

        self._d_params_xc = grad_mat

        gt = grad_mat.dot(grad_loss_params)
        return gt.ravel()

    def _rnd_init_poisoning_points(
            self, n_points=None, init_from_val=False, val=None):
        """Returns a random set of poisoning points randomly with
        flipped labels."""
        if init_from_val:
            if val:
                init_dataset = val
            else:
                init_dataset = self.val
        else:
            init_dataset = self.training_data

        if (self.n_points is None or self.n_points == 0) and (
                n_points is None or n_points == 0):
            raise ValueError("Number of poisoning points (n_points) not set!")

        if n_points is None:
            n_points = self.n_points

        idx = CArray.randsample(init_dataset.num_samples, n_points,
                                random_state=self.random_seed)

        xc = init_dataset.X[idx, :].deepcopy()

        # if the attack is in a continuous space we add a
        # little perturbation to the initial poisoning point
        random_noise = CArray.rand(shape=xc.shape,
                                   random_state=self.random_seed)
        xc += 1e-3 * (2 * random_noise - 1)
        yc = CArray(init_dataset.Y[idx]).deepcopy()  # true labels

        # randomly pick yc from a different class
        for i in range(yc.size):
            labels = CArray.randsample(init_dataset.num_classes, 2,
                                       random_state=self.random_seed)
            if yc[i] == labels[0]:
                yc[i] = labels[1]
            else:
                yc[i] = labels[0]

        return xc, yc


    def _run(self, xc, yc, idx=0):
        """Single point poisoning.

        Here xc can be a *set* of points, in which case idx specifies which
        point should be manipulated by the poisoning attack.

        """
        xc = CArray(xc.deepcopy()).atleast_2d()

        self._yc = yc
        self._xc = xc
        self._idx = idx  # point to be optimized within xc

        self._x0 = self._xc[idx, :].ravel()

        self._init_solver()

        if self.y_target is None:  # indiscriminate attack
            x = self._solver.maximize(self._x0)
        else:  # targeted attack
            x = self._solver.minimize(self._x0)

        self._solution_from_solver()

        return x

    def run(self, x, y, ds_init=None, max_iter=1):
        """Runs poisoning on multiple points.

        It reads n_points (previously set), initializes xc, yc at random,
        and then optimizes the poisoning points xc.

        Parameters
        ----------
        x : CArray
            Validation set for evaluating classifier performance.
            Note that this is not the validation data used by the attacker,
            which should be passed instead to `CAttackPoisoning` init.
        y : CArray
            Corresponding true labels for samples in `x`.
        ds_init : CDataset or None, optional.
            Dataset for warm start.
        max_iter : int, optional
            Number of iterations to re-optimize poisoning data. Default 1.

        Returns
        -------
        y_pred : predicted labels for all val samples by targeted classifier
        scores : scores for all val samples by targeted classifier
        adv_xc : manipulated poisoning points xc (for subsequents warm starts)
        f_opt : final value of the objective function

        """
        if self.n_points is None or self.n_points == 0:
            # evaluate performance on x,y
            y_pred, scores = self.classifier.predict(
                x, return_decision_function=True)
            return y_pred, scores, ds_init, 0

        # n_points > 0
        if self.init_type == 'random':
            # randomly sample xc and yc
            xc, yc = self._rnd_init_poisoning_points()
        else:
            raise NotImplementedError(
                "Unknown poisoning point initialization strategy.")

        # re-set previously-optimized points if passed as input
        if ds_init is not None:
            xc[0:ds_init.num_samples, :] = ds_init.X
            yc[0:ds_init.num_samples] = ds_init.Y

        delta = 1.0
        k = 0

        # max_iter ignored for single-point attacks
        if self.n_points == 1:
            max_iter = 1

        metric = CMetric.create('accuracy')

        while delta > 0 and k < max_iter:

            # self.logger.info(
            #     "Iter on all the poisoning samples: {:}".format(k))

            xc_prv = xc.deepcopy()
            for i in range(self.n_points):
                # this is to optimize the last points first
                # (and then re-optimize the first ones)
                idx = self.n_points - i - 1
                xc[idx, :] = self._run(xc, yc, idx=idx)
                # optimizing poisoning point 0
                # self.logger.info(
                #     "poisoning point {:} optim fopt: {:}".format(
                #         i, self._f_opt))

                y_pred, scores = self._poisoned_clf.predict(
                    x, return_decision_function=True)
                acc = metric.performance_score(y_true=y, y_pred=y_pred)
                # self.logger.info("Poisoned classifier accuracy "
                #                  "on test data {:}".format(acc))

            delta = (xc_prv - xc).norm_2d()
            # self.logger.info(
            #     "Optimization with n points: " + str(self.n_points) +
            #     " iter: " + str(k) + ", delta: " +
            #     str(delta) + ", fopt: " + str(self._f_opt))
            k += 1

        # re-train the targeted classifier (copied) on poisoned data
        # to evaluate attack effectiveness on targeted classifier
        clf, tr = self._update_poisoned_clf(clf=self.classifier,
                                            tr=self._training_data,
                                            train_normalizer=False)
        # fixme: rechange train_normalizer=True

        y_pred, scores = clf.predict(x, return_decision_function=True)
        acc = metric.performance_score(y_true=y, y_pred=y_pred)
        # self.logger.info(
        #     "Original classifier accuracy on test data {:}".format(acc))

        return y_pred, scores, CDataset(xc, yc), self._f_opt


if __name__ == '__main__':

    mnist = fetch_openml('mnist_784')
    X, y = mnist["data"], mnist["target"]

    X = np.array(X)
    y = np.array(y)
    y = y.astype(np.uint8)

    target_digit1 = 1
    target_digit2 = 7

    target_digit1_xdata = X[y == target_digit1]
    target_digit2_xdata = X[y == target_digit2]
    target_digit1_ydata = y[y == target_digit1]
    target_digit2_ydata = y[y == target_digit2]

    X2 = np.concatenate((target_digit1_xdata, target_digit2_xdata), axis=0)
    y2 = np.concatenate((target_digit1_ydata, target_digit2_ydata), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.6, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    svm_clf = SVC(kernel='linear', C=1.0, random_state=42)
    svm_clf.fit(X_train, y_train)

    # <a id='part3.3'></a>
    # ### 3.3: SVM Model Prediction

    # In[14]:


    y_pred = svm_clf.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print('Accuracy: %.3f' % (accuracy))
    print('Precision: %.3f' % (precision))
    print('Recall: %.3f' % (recall))
    print(cm)