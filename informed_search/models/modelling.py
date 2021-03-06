
"""
Author:         Nemanja Rakicevic
Date  :         January 2018
Description:
                Classes containing different search algorithm implementations
                and auxiliary functions to interface the learned model.
                Search algorithms:
                - informed search
                - uidf-based
                - entropy
                - bayesian optimisation based
                - random
"""

import pdb

import os
import logging
import pickle
import itertools
import numpy as np

from scipy.stats import norm
from numpy.core.umath_tests import inner1d
from heapq import nlargest, nsmallest
from functools import partial

import informed_search.utils.plotting as uplot
import informed_search.models.kernels as kern

from informed_search.utils.misc import _EPS, _TAB
from informed_search.utils.misc import elementwise_sqdist, scaled_sqdist


logger = logging.getLogger(__name__)


class BaseModel(object):
    """Base task model learning class."""

    def __init__(self,
                 seed,
                 dirname,
                 verbose,
                 parameter_list,
                 kernel_name,
                 kernel_lenscale,
                 kernel_sigma,
                 pidf_coeff,
                 **kwargs):
        # Fix numpy seed
        np.random.seed(seed)
        # Generate experiment directory
        self.dirname = dirname
        self.verbose = verbose
        # Initialise parameter space
        self.param_list = parameter_list
        self.param_space = np.array([xs for xs
                                    in itertools.product(*self.param_list)])
        self.param_dims = tuple([len(i) for i in self.param_list])
        self.n_coords = len(self.param_space)
        self.n_param = len(self.param_list)
        # Initialise statistics
        self.coord_explored = []
        self.coord_failed = []
        # Initialise attributes
        self.name = self.__class__.__name__
        self._build_model(pidf_coeff)
        self._build_kernel(kernel_name, kernel_lenscale, kernel_sigma)

    def _build_model(self, pidf_coeff):
        """Initialise task model and exploration components."""
        # Prior distribution
        self.prior_init = np.ones(self.param_dims) / np.product(self.param_dims)
        # Create covariance matrix
        if pidf_coeff is not None:
            self.pidf_coeff = pidf_coeff
            self.pidf_cov = self.pidf_coeff * np.eye(self.n_param)
        # Initialise informed search components
        self.pidf = self.prior_init
        self.uidf = self.prior_init
        self.sidf = self.prior_init  # np.ones(tuple(self.param_dims))
        # Initialise task model components
        self.model_list = []
        self.mu_alpha = np.zeros(self.param_dims)
        self.mu_L = np.zeros(self.param_dims)
        self.var_alpha = np.ones(self.param_dims)
        self.var_L = np.ones(self.param_dims)
        # Initialise additional components
        self.uidf = np.ones(self.param_dims)
        self.mu_pidf = 0.5 * np.ones(self.param_dims)
        self.var_pidf = np.ones(self.param_dims)
        self.mu_uncert = 0 * np.ones(self.param_dims)
        self.var_uncert = np.ones(self.param_dims)
        self.previous_uncertainty = self.uncertainty
        self.delta_uncertainty = []

    def _build_kernel(self, kernel_name, kernel_lenscale, kernel_sigma):
        """Construct the kernel function and initialise Kss."""
        self.kernel_fn = partial(getattr(kern, kernel_name + '_kernel'),
                                 kernel_lenscale=kernel_lenscale,
                                 kernel_sigma=kernel_sigma)
        # self.Kss = self.kernel_fn(a=self.param_space, b=self.param_space)
        self.Kss_diag = kernel_sigma * np.ones(len(self.param_space))

    def query_target(self, angle_s, dist_s):
        """Generate test parameter coordinates for given target polar coords."""
        thrsh = 0.1
        # Calculate model error for target point
        m_meas = elementwise_sqdist(self.mu_alpha - angle_s, self.mu_L - dist_s)
        # m_meas  = scaled_sqdist(M_angle, M_dist, angle_s, dist_s)
        # Generate optimal parameters to reached the target point
        src = 0
        for cnt in range(len(m_meas.ravel())):
            src += 1
            # get candidate movement parameter vector with smalles model error
            best_fit = nsmallest(src, m_meas.ravel())
            selected_coord = np.argwhere(m_meas == best_fit[src - 1])[0]
            selected_params = np.array([self.param_list[i][selected_coord[i]]
                                       for i in range(self.n_param)])
            selected_mu_alpha = self.mu_alpha[tuple(selected_coord)]
            selected_mu_L = self.mu_L[tuple(selected_coord)]
            error_angle = selected_mu_alpha - angle_s
            error_dist = selected_mu_L - dist_s
            if cnt % 50 == 0 and cnt > 100:
                thrsh = cnt / 1000.
                src = 0
            # check pidf constraint is below threshold (if close to failure)
            pidf_value = self.pidf[tuple(selected_coord)]
            uidf_value = self.uidf[tuple(selected_coord)]
            sidf_value = self.sidf[tuple(selected_coord)]
            if pidf_value < thrsh:
                break
            else:
                # continue to the next smallest model error
                if self.verbose & 2:
                    logger.info(
                        "(Iteration: {}; search {}) "
                        "--- BAD SAMPLE, resampling ..."
                        "\n{} - PIDF: {:4.2f}; UIDF: {:4.2f}; SIDF: {:4.2f};"
                        "\n{} - threshold: {}".format(
                            cnt, src, _TAB, pidf_value, uidf_value, sidf_value,
                            _TAB, thrsh))
        if self.verbose & 2:
            logger.info("(Iteration: {})"
                "\n{} - PIDF: {:4.2f}; UIDF: {:4.2f}; SIDF: {:4.2f};"
                "\n{} - Generated coords: {} -> parameters: {}".format(
                    cnt, _TAB, pidf_value, uidf_value, sidf_value,
                    _TAB, selected_coord, selected_params))
        # Return vector to execute as well as estimated model polar error
        return selected_coord, selected_params, \
            (selected_mu_alpha, selected_mu_L), best_fit[src - 1], pidf_value

    def generate_sample(self, info_list=None, **kwargs):
        """
        Generate the movement parameter vector to evaluate next,
        based on the calculated SIDF.
        """
        # Check if the parameters have already been used
        temp_good = np.array([])
        cnt = 1
        while len(temp_good) == 0:
            sample = np.array([self.sidf == c for c
                              in nlargest(cnt * 1, self.sidf.ravel())])
            sample = sample.reshape([-1] + list(self.param_dims))
            sample_idx = np.argwhere(sample)[:, 1:]
            temp_good = np.array(list(set(map(tuple, sample_idx)) -
                                 set(map(tuple, self.coord_explored))))
            cnt += 1
            if cnt > self.n_coords:
                logger.info("All parameters have been explored!\tEXITING...")
                return None, None
        # Convert from coordinates to parameters
        selected_coord = temp_good[np.random.choice(len(temp_good)), :]
        selected_params = np.array([self.param_list[i][selected_coord[i]]
                                    for i in range(self.n_param)])
        self.coord_explored.append(selected_coord)
        # Return the next parameter vector
        return selected_coord, selected_params

    def update_GPR(self, xtrain, ytrain=None):
        """Update Gaussian Process Regression model over the parameter set."""
        xtest = self.param_space
        # Calculate kernel matrix of training points
        K = self.kernel_fn(xtrain, xtrain)
        # Get the matrix square root via Cholesky decomposition
        L = np.linalg.cholesky(K + _EPS * np.eye(len(xtrain)))
        # Calculate train-test kernel matrix
        Ks = self.kernel_fn(xtrain, xtest)
        Lk = np.linalg.solve(L, Ks)
        # Get variance posterior
        var_post = np.sqrt(self.Kss_diag - np.sum(Lk**2, axis=0))
        # Update just uncertainty or model GPR
        if ytrain is None:
            return var_post.reshape(self.param_dims)
        else:
            # Get mean posterior
            mu_post = np.dot(Lk.T, np.linalg.solve(L, ytrain))
            return mu_post.reshape(self.param_dims), \
                var_post.reshape(self.param_dims)  # /np.sum(var_post)

    def save_model(self, num_trial, save_plots=True, save_data=True, **kwargs):
        """Save model data as checkpoints."""
        if save_plots:
            uplot.plot_model(
                model_object=self,
                dimensions=[0, 1],
                num_trial=num_trial,
                savepath=self.dirname)
        if save_data:
            self.model_list.append([
                self.mu_alpha,
                self.mu_L,
                self.uidf,
                self.pidf,
                self.sidf,
                self.param_list])
            with open(self.dirname + "/experiment_dataset.dat", "wb") as f:
                pickle.dump(self.model_list, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            # Avoid crashing during writing
            os.system("mv {} {}".format(
                self.dirname + "/experiment_dataset.dat",
                self.dirname + "/experiment_dataset.pkl"))

    def load_model(self, loadpath):
        """Load model data from checkpoint."""
        logger.info("Loading: {}".format(loadpath))
        filename = os.path.join(self.dirname, "experiment_dataset.pkl")
        with open(filename, "rb") as f:
            self.model_list = pickle.load(f)
            self.mu_alpha, \
                self.mu_L, \
                self.uidf, \
                self.pidf, \
                self.sidf, \
                self.param_list = self.model_list[-1]

    @property
    def uncertainty(self):
        """Return model uncertainty."""
        return self.uidf.mean()

    @property
    def components(self):
        """Return model components."""
        return self.mu_alpha, self.mu_L, self.var_alpha, self.var_L

    def update_model():
        raise NotImplementedError


class InformedSearch(BaseModel):
    """
    Proposed Informed Search model class;
    Uses the penalisation and uncertainty IDF's to determine the next trial.
    """

    def update_model(self, info_list, save_model_progress=False, **kwargs):
        """
        Select successful trials to estimate the GPR model's mean and
        variance, and the failed ones to update the penalisation IDF.
        """
        if len(info_list):
            # Update task models
            if info_list[-1]['fail_status'] == 0:
                good_params = np.vstack([tr['parameters'] for tr in info_list
                                        if tr['fail_status'] == 0])
                good_fevals = np.vstack([tr['ball_polar'] for tr in info_list
                                        if tr['fail_status'] == 0])
                self.mu_alpha, self.var_alpha = self.update_GPR(
                    good_params, good_fevals[:, 0])
                self.mu_L, self.var_L = self.update_GPR(
                    good_params, good_fevals[:, 1])
                self.update_PIDF(info_list[-1]['parameters'], failed=False)
            # Update PIDF
            elif info_list[-1]['fail_status'] > 0:
                self.coord_failed = np.array([tr['coordinates'] for tr
                                             in info_list
                                             if tr['fail_status'] > 0])
                self.update_PIDF(info_list[-1]['parameters'], failed=True)
            # Update UIDF
            all_trials = np.array([tr['parameters'] for tr in info_list])
            self.uidf = self.update_GPR(all_trials, None)
            # Update SIDF using model uncertainty and the penalisation IDF
            sidf = 1.0 * self.uidf * (1 - self.pidf)
            self.sidf = sidf / (sidf.max() + _EPS)

            if save_model_progress:
                self.save_model(**kwargs)

    def generate_pdf_matrix(self, x_sample, mu, cov):
        """Create a multivariate Gaussian over the parameter space."""
        tmp = np.dot((x_sample - mu), np.linalg.inv(cov))
        tmp_t = (x_sample - mu).T
        denom = np.sqrt(2 * np.pi * np.linalg.det(cov))
        f = (1 / denom) * np.exp(-0.5 * inner1d(tmp, tmp_t.T))
        return f

    def update_PIDF(self, mu, failed=1):
        """
        Update the PIDF based on the failed trials, by centering a
        negative multivariate Gaussian on the point in the parameter space
        corresponding to the movement vector of the failed trial.

        For successful trials a positive Gaussian with a larger variance
        is used.
        """
        previous_pidf = self.pidf.copy()
        # Generate the trial Gaussian, depending on outcome
        if failed:
            # scale covariance matrix based on failure frequency
            fcnt = np.array([max(np.bincount(self.coord_failed[:, f]))
                             for f in range(self.n_param)], np.float)
            cov_diag = 1 + (fcnt - fcnt.min()) / fcnt.max()
        else:
            # fixed covariance matrix for successful trial outcomes
            cov_diag = 0.5 * np.ones(self.n_param)
        np.fill_diagonal(self.pidf_cov, self.pidf_coeff * cov_diag)
        # Estimate the contributing Gaussian
        trial_gaussian = np.reshape(
            self.generate_pdf_matrix(self.param_space, mu, self.pidf_cov),
            self.param_dims)
        trial_gaussian /= (trial_gaussian.max() + _EPS)
        trial_gaussian *= 1 if failed else -1
        # Update the penalisation IDF
        self.pidf = np.clip(previous_pidf + trial_gaussian, self.prior_init, 1.)


class UIDFSearch(BaseModel):
    """
    Uncertainty-only model class;
    Uses only the uncertainty IDF to select the next trial. Does not update
    the penalisation IDF.
    """

    def update_model(self, info_list, save_model_progress=False, **kwargs):
        """
        Select successful trials to estimate the GPR model mean and variance,
        and the failed ones to update the penalisation IDF.
        """
        if len(info_list):
            # Update task models
            if info_list[-1]['fail_status'] == 0:
                good_params = np.vstack([tr['parameters'] for tr in info_list
                                        if tr['fail_status'] == 0])
                good_fevals = np.vstack([tr['ball_polar'] for tr in info_list
                                        if tr['fail_status'] == 0])
                # Update the Angle and Distance GPR models
                self.mu_alpha, self.var_alpha = self.update_GPR(
                    good_params, good_fevals[:, 0])
                self.mu_L, self.var_L = self.update_GPR(
                    good_params, good_fevals[:, 1])
            elif info_list[-1]['fail_status'] > 0:
                self.coord_failed = np.array([tr['coordinates'] for tr
                                             in info_list
                                             if tr['fail_status'] > 0])
            # Update UIDF
            all_trials = np.array([tr['parameters'] for tr in info_list])
            self.uidf = self.update_GPR(all_trials, None)
            # Update SIDF uisng only model uncertainty
            sidf = 1.0 * self.uidf
            self.sidf = sidf / (sidf.max() + _EPS)

            if save_model_progress:
                self.save_model()


class EntropySearch(BaseModel):
    """
    Entropy-based model class;
    Uses the entropy of uncertainty IDF to select the next trial.
    Does not update the penalisation IDF.
    """

    def update_model(self, info_list, save_model_progress=False, **kwargs):
        """
        Use successful trials to estimate the GPR model mean, and
        all trials to estimate the UIDF.
        """
        if len(info_list):
            # Update task models
            if info_list[-1]['fail_status'] == 0:
                good_params = np.vstack([tr['parameters'] for tr in info_list
                                        if tr['fail_status'] == 0])
                good_fevals = np.vstack([tr['ball_polar'] for tr in info_list
                                        if tr['fail_status'] == 0])
                self.mu_alpha, self.var_alpha = self.update_GPR(
                    good_params, good_fevals[:, 0])
                self.mu_L, self.var_L = self.update_GPR(
                    good_params, good_fevals[:, 1])
            elif info_list[-1]['fail_status'] > 0:
                self.coord_failed = np.array([tr['coordinates'] for tr
                                             in info_list
                                             if tr['fail_status'] > 0])
            # Update UIDF
            all_trials = np.array([tr['parameters'] for tr in info_list])
            self.uidf = self.update_GPR(all_trials, None)
            # Update SIDF using the posterior distributions entropy
            sidf = 0.5 * np.log(2 * np.pi * np.e * self.uidf)
            self.sidf = sidf / (sidf.max() + _EPS)

            if save_model_progress:
                self.save_model()


class BOSearch(BaseModel):
    """
    A modified BO approach based model class;
    adapted from Englert and Toussaint (2016) to next trial selection.
    This comparison was proposed by the reviewer.
    """

    def update_model(self, info_list, save_model_progress=False, **kwargs):
        """
        Use successful trials to estimate the GPR model mean,
        all trials to estimate the Bayesian Optimisation components.
        """
        if len(info_list):
            # Update task models
            if info_list[-1]['fail_status'] == 0:
                good_params = np.vstack([tr['parameters'] for tr in info_list
                                        if tr['fail_status'] == 0])
                good_fevals = np.vstack([tr['ball_polar'] for tr in info_list
                                        if tr['fail_status'] == 0])
                self.mu_alpha, self.var_alpha = self.update_GPR(
                    good_params, good_fevals[:, 0])
                self.mu_L, self.var_L = self.update_GPR(
                    good_params, good_fevals[:, 1])
            elif info_list[-1]['fail_status'] > 0:
                self.coord_failed = np.array([tr['coordinates'] for tr
                                             in info_list
                                             if tr['fail_status'] > 0])
            # Update UIDF
            all_trials = np.array([tr['parameters'] for tr in info_list])
            self.uidf = self.update_GPR(all_trials, None)
            # BO pidf component
            fail_indicator = np.array([1 if tr['fail_status'] > 0 else 0 for tr
                                      in info_list])
            self.mu_pidf, self.var_pidf = self.update_GPR(
                all_trials, fail_indicator)
            # BO uncertainty component
            temp = self.uncertainty
            self.delta_uncertainty.append(self.previous_uncertainty - temp)
            self.previous_uncertainty = temp
            assert len(self.delta_uncertainty) == len(info_list)
            self.mu_uncert, self.var_uncert = self.update_GPR(
                all_trials, np.array(self.delta_uncertainty))
            # Update SIDF
            sidf = np.zeros(self.param_dims)
            # Calculate only PI for samples classified as successful
            if len(self.delta_uncertainty) > 0:
                diff_tmp = self.mu_uncert \
                    - np.array(self.delta_uncertainty).max()
                part_PI = norm.cdf( diff_tmp / np.sqrt(self.var_uncert))
            else:
                part_PI = norm.cdf((self.mu_uncert) / np.sqrt(self.var_uncert))
            part_PI[part_PI <= 0.5] = 0.
            part_Vgs = self.var_pidf.copy()
            part_Vgs[self.mu_pidf != 0.5] = 0
            sidf += part_PI
            sidf += part_Vgs
            self.sidf = sidf

            if save_model_progress:
                self.save_model()


class RandomSearch(BaseModel):
    """
    Random search based class;
    Performs random search in the parameter space, without replacement.
    """

    def update_model(self, info_list, save_model_progress=False, **kwargs):
        """
        Use successful trials to estimate the GPR model mean and variance.
        Random search does not use any exploration components.
        """
        if len(info_list):
            # Update task models
            if info_list[-1]['fail_status'] == 0:
                good_params = np.vstack([tr['parameters'] for tr in info_list
                                        if tr['fail_status'] == 0])
                good_fevals = np.vstack([tr['ball_polar'] for tr in info_list
                                        if tr['fail_status'] == 0])
                self.mu_alpha, self.var_alpha = self.update_GPR(
                    good_params, good_fevals[:, 0])
                self.mu_L, self.var_L = self.update_GPR(
                    good_params, good_fevals[:, 1])
            elif info_list[-1]['fail_status'] > 0:
                self.coord_failed = np.array([tr['coordinates'] for tr
                                             in info_list
                                             if tr['fail_status'] > 0])

            if save_model_progress:
                self.save_model()

    def generate_sample(self, *args, **kwargs):
        """
        Generate the random movement parameter vector to evaluate next.
        (without replacement)
        """
        param_sizes = [range(i) for i in self.param_dims]
        temp = np.array([xs for xs in itertools.product(*param_sizes)])
        # Check if the parameters have already been used
        temp_good = np.array([])
        cnt = 1
        while len(temp_good) == 0:
            sample = np.array([temp[np.random.choice(len(temp)), :]])
            temp_good = np.array(list(set(map(tuple, sample)) -
                                 set(map(tuple, self.coord_explored))))
            cnt += 1
            if cnt > self.n_coords:
                logger.info("All parameters have been explored!\tEXITING...")
                return None, None
        # Convert from coordinates to parameters
        selected_coord = temp_good[np.random.choice(len(temp_good)), :]
        selected_params = np.array([self.param_list[i][selected_coord[i]]
                                    for i in range(self.n_param)])
        self.coord_explored.append(selected_coord)
        # return the next sample vector
        return selected_coord, selected_params
