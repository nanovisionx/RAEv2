import torch as th
import torch.nn.functional as F

from stage2.utils import apply_cfg_dropout


def _expand_t(t, x):
    return t.view(t.size(0), *([1] * (len(x.size()) - 1)))


def get_time_sampler(time_dist_type: str):
    parts = time_dist_type.split("_")
    name = parts[0]
    if name == "logit-normal":
        assert len(parts) == 3, f"Expected 'logit-normal_MU_SIGMA', got '{time_dist_type}'"
        mu, sigma = float(parts[1]), float(parts[2])
        assert sigma > 0, "sigma must be > 0"
        return lambda bs: (th.randn(bs) * sigma + mu).sigmoid()
    else:
        raise NotImplementedError(f"Unknown time distribution: {time_dist_type}")


class Transport:
    def __init__(self, prediction="velocity", time_dist_type="logit-normal_0_1", time_dist_shift=1.0, t_eps=0.05):
        self.prediction = prediction
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        self.t_eps = t_eps
        self.time_sampler = get_time_sampler(time_dist_type)

    def sample(self, x1):
        x0 = th.randn_like(x1)
        t = self.time_sampler(x1.shape[0]).to(x1)
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t, x0, x1

    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################
    def training_losses(self, model, x1, model_kwargs={}, model_kwargs_null={}, z_clean=None, repa_coeff=None, base_model_coeff=1.0, cfg_dropout_prob=0.1):
        model_kwargs, _ = apply_cfg_dropout(model_kwargs, model_kwargs_null, cfg_dropout_prob)

        t, x0, x1 = self.sample(x1)
        xt = (1 - _expand_t(t, x1)) * x1 + _expand_t(t, x1) * x0
        vt = (xt - x1) / _expand_t(t, xt).clamp_min(self.t_eps)

        enable_repa = z_clean is not None and repa_coeff is not None
        zt_pred = None
        if enable_repa:
            model_output, zt_pred = model(xt, t, return_intermediate=True, **model_kwargs)
        else:
            model_output = model(xt, t, **model_kwargs)

        # Handle Internal Guidance dual-output models (full, base)
        base_output = None
        if isinstance(model_output, tuple) and len(model_output) == 2:
            model_output, base_output = model_output

        terms = {'loss': self.compute_loss(model_output, vt, xt, t)}
        if base_output is not None:
            loss_base = self.compute_loss(base_output, vt, xt, t)
            terms['loss'] = terms['loss'] + base_model_coeff * loss_base
            terms['loss_base'] = loss_base
        if enable_repa and zt_pred is not None:
            terms['loss_repa'] = repa_coeff * F.mse_loss(zt_pred, z_clean)

        return terms

    def convert_model_pred(self, output, xt, t):
        # Unify model output to v-pred
        if self.prediction == "velocity":
            return output
        elif self.prediction == "x":
            t_safe = _expand_t(t, xt).clamp_min(self.t_eps)
            return (xt - output) / t_safe

    def compute_loss(self, output, vt, xt, t):
        output = self.convert_model_pred(output, xt, t)
        return (output - vt) ** 2

    def get_drift(self):
        def body_fn(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            return self.convert_model_pred(model_output, x, t)
        return body_fn
