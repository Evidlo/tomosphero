"""Optimizer loss functions

The classes in this module serve as loss functions for the iterative reconstruction methods in `retrieval.py`.

The class should be initialized by the user and then passed to the reconstruction method which will call `.compute()`
on the loss function and try to minimize the weighted sum of all provided losses.

Weights are stored internally in the `lam` parameter, which the user may set by multiplying the
initialized loss object with a float or by providing a `lam` kwarg on initialization.
"""

import torch as t

class Loss:
    """Base class for tomographic retrieval loss function terms

    Subclass this class and implement `compute(…)` to create new loss function terms.
    Instantiating Loss objects may be multiplied by a float to set weight (see Usage).

    Args:
        projection_mask (tensor): column densities to mask out when computing loss
        obj_mask (tensor, optional): voxels to mask out when computing loss
        lam (float, optional): loss function scaling (default 1)
        use_grad (bool, optional): whether this loss function's gradient needs to be
            used in optimization (default True)

    Usage:
    ``` python
        gd(..., loss_fns=[5 * MyLoss(), 3 * MyLoss2()], ...)
    ```

    Attributes:
        kind (str): Category of loss - one of either 'fidelity', 'regularizer', or 'oracle'.
            - fidelity - term is data-fidelity term
            - regularizer - term is a regularizer
            - oracle - term is for logging purposes, not used in loss function
            This property is used by `loss_plot(…)` for display purposes.
            Function `gd(…)` also uses this property when generating loss term summary
            in the TQDM status bar.
        lam (float): weight hyperparameter for this term

    """

    kind = 'regularizer'

    def __init__(
            self, *args, projection_mask=1, obj_mask=1, lam=1,
            use_grad=True, **kwargs
        ):
        """@private"""
        self.projection_mask = projection_mask
        self.obj_mask = obj_mask
        self.lam = lam
        self.use_grad = use_grad

    def compute(self, f, y, d, c):
        """Compute loss

        Args:
            f (Operator): forward function. density→projections
            y (tensor): measurements.  shape must match `projection_mask`
            d (tensor): density to pass through forward function.
                shape must match `obj_mask`
            c (tensor): coefficients of shape model.coeffs_shape

        Returns:
            loss (float)
        """
        raise NotImplemented

    def __call__(self, f, y, d, c):
        """Compute loss, incorporating loss weight and whether pytorch grad is needed

        Args:
            f (Operator): forward function. density→projections
            y (tensor): measurements.  shape must match `projection_mask`
            d (tensor): density to pass through forward function.
                shape must match `obj_mask`
            c (tensor): coefficients of shape model.coeffs_shape

        Returns:
            loss (float or None)
        """
        if self.use_grad:
            result = self.compute(f, y, d, c)
        else:
            with t.no_grad():
                result = self.compute(f, y, d, c)
        return None if result is None else self.lam * result

    def __mul__(self, other):
        """Allow multiplying Loss object with scalar hyperparameter"""
        self.lam = other
        return self

    def __rmul__(self, other):
        """Allow multiplying Loss object with scalar hyperparameter"""
        return self.__mul__(other)

    def __repr__(self):
        return f'{self.lam:.0e} * {type(self).__name__}'
        # return f'{type(self).__name__}'


class SquareLoss(Loss):
    """Standard mean L2 loss"""

    kind = 'fidelity'

    def compute(self, f, y, d, c):
        """"""
        result = t.mean(self.projection_mask * (y - f(d * self.obj_mask))**2)
        return result


class SquareRelLoss(Loss):
    """Loss as mean percent error"""

    kind = 'fidelity'

    def compute(self, f, y, d, c):
        """"""
        obs = f(d * self.obj_mask)

        # rel_err = (y - obs) / y
        # rel_err = rel_err.nan_to_num() * self.projection_mask

        zero_mask = (y != 0)
        rel_err = t.zeros_like(y)
        rel_err[zero_mask] = (y - obs)[zero_mask] / y[zero_mask]

        return t.mean((self.projection_mask * rel_err)**2)


class AbsLoss(Loss):
    """Mean L1 loss"""

    kind = 'fidelity'

    def compute(self, f, y, d, c):
        """"""
        result = t.mean(self.projection_mask * (y - f(d * self.obj_mask)).abs())
        return result


class CheaterLoss(Loss):
    """L2 loss directly over density ground truth"""

    # do not use for gradient descent
    kind = 'oracle'

    def __init__(self, density_truth, *args, **kwargs):
        """Setup loss

        Args:
            density_truth (tensor): ground truth density
            *args: position args passed to Loss
            **kwargs: keyword args passed to Loss
        """

        self.density_truth = density_truth
        super().__init__(**kwargs)

    def compute(self, f, y, d, c):
        """"""
        return t.mean(self.obj_mask * (d - self.density_truth)**2)


class NegRegularizer(Loss):
    """Mean of negative voxels"""

    def compute(self, f, y, d, c):
        """"""
        return t.mean(t.abs(self.obj_mask * d.clip(max=0)))


class NegSumRegularizer(Loss):
    """Sum of negative voxels"""
    def compute(self, f, y, d, c):
        """"""
        return t.sum(t.abs(self.obj_mask * d.clip(max=0)))


class RelError(Loss):
    """Error relative to ground truth.

    Used only for plotting purposes and does not impact loss in `gd(…)`"""

    # do not use for gradient descent
    kind = 'oracle'

    def __init__(self, truth, grid, mode='max', interval=10, **kwargs):
        """Setup loss

        Args:
            truth (tensor): ground truth object to compare against
            grid (sph_raytracer SphericalGrid): truth spherical grid
            mode (str): Reduction method to compute single scalar. Either 'mean' or 'max'
            interval (int): compute result every `interval` iterations to save compute time
        """
        self.use_grad = False
        self.truth = truth
        self.interval = interval
        self.mode = mode
        self.n = 0
        self.grid = grid

        super().__init__(**kwargs)

    def compute(self, f, y, d, c):
        """
        Args:
            f (function): forward operator from density to measurements
            y (tensor): actual noisy measurements
            d (tensor): candidate reconstructed density
            c (tensor): coefficients for model.  low-rank representation of
                density

        Returns:
            loss (float)
        """
        # only perform computation every `interval` calls to save on compute time
        # otherwise, return NaN
        if self.n % self.interval == 0:
            rel_err = t.abs(self.truth - d) / self.truth
            if self.mode == 'mean':
                result = rel_err[self.truth_cart > 25].mean()
            elif self.mode == 'max':
                result = rel_err[self.truth_cart > 25].max()
            else:
                raise ValueError("Invalid mode")
        else:
            result = float('nan')

        self.n += 1
        return result


class ReqErrOld(Loss):
    """25 atoms/cc region mean absolute percent error"""

    def __init__(self, density_truth, mode='max', interval=10, **kwargs):
        """Setup loss

        Args:
            density_truth (tensor): ground truth density to compare against
            mode (str): Collapse relative error for all voxels into single value. Either 'mean' or 'max'
            interval (int): compute result every `interval` iterations to save compute time
        """
        self.use_grad = False
        self.density_truth = density_truth
        self.interval = interval
        self.mode = mode
        self.n = 0

        super().__init__(**kwargs)

    def compute(self, f, y, d, c):
        """"""
        if self.n % self.interval == 0:
            rel_err = t.abs(self.density_truth - d) / self.density_truth
            if self.mode == 'mean':
                result = rel_err[self.density_truth > 25].mean()
            elif self.mode == 'max':
                result = rel_err[self.density_truth > 25].max()
            else:
                raise ValueError("Invalid mode")
        else:
            result = float('nan')

        self.n += 1
        return result


class IRLSLoss(Loss):
    """Iteratively reweighted least-squares loss
    """

    kind = 'fidelity'

    @classmethod
    def noise_var(self, instruments, bin_funcs):
        """Get noise variances for science binned images

        Args:
            instruments (list[Instrument]): instruments which have LOS noise variances
            bin_funcs (list[SciencePixelBinning] or None): binning functions

        Returns:
            var (tensor): 3D array of noise variances
        """
        # variances of science pixels
        sci_var = []
        for i, b in tqdm(zip(instruments, bin_funcs)):
            # FIXME: inline modification of i.scene because `instrument_cal` expects mean_oob
            i.scene.mean_oob = 0
            rect_var = i.calibrate_variance(i.var_y)
            pixcnt = b.get_pix_count()
            # variance of a mean → divide by N²
            sci_var.append(b(rect_var, b.get_pix_count()**2))

        return t.from_numpy(np.stack(sci_var))

    def __init__(self, noise_var, *args, **kwargs):
        """Setup loss

        Args:
            noise_var (tensor): variance of noise.  shape should be equal to y
            *args: position args passed to Loss
            **kwargs: keyword args passed to Loss
        """

        self.noise_var = noise_var
        super().__init__(**kwargs)

        self.n = 0

    def compute(self, f, y, d, c):
        """"""
        self.n += 1

        # absolute measurement residuals
        res_abs = self.projection_mask * (y - f(d * self.volume_mask)).abs()
        weight = 1 / (self.noise_var + res_abs**2)
        # weight = 1
        weight = 1
        result = t.mean(weight * res_abs)
        # if self.n % 100 == 0:
        #     import ipdb
        #     ipdb.set_trace()
        return result

class SphHarmL1Regularizer(Loss):
    """Compute loss with custom function"""

    kind = 'regularizer'

    def __init__(self, model, k=2.5, device=None, **kwargs):
        """
        Args:
            model (SphHarmSplineModel): specific sph. harm. spline model
            k (float): exponential decay to normalize coefficients
            device (None, str, or torch.device): device for returned array
        """
        self.model = model
        self.k = k
        self.device = device

        super().__init__(**kwargs)

    def compute(self, f, y, d, c):
        """"""
        # x = self.model.sph_coeffs(c) / self.model.grid.r.to(self.model.device)**-self.k
        # normalize by A00 term
        x = self.model.sph_coeffs(c) / self.model.sph_coeffs(c)[..., 0, :]
        # x = self.model.sph_coeffs(c) / self.model.sph_coeffs(c)[..., 0, 0]
        return x.abs().mean()