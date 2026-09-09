"""Run from the repository root: python -m tests.check_layernorm."""

import argparse

import numpy as np
import torch

from aif.nn.normalizations import LayerNorm2D


class CompatibleArray(np.ndarray):
    """Diagnostic only: translate ndarray.var(correction=...) to ddof=...."""

    def var(self, *args, **kwargs):
        if "correction" in kwargs:
            kwargs["ddof"] = kwargs.pop("correction")
        return super().var(*args, **kwargs)


def compare(name, x, gamma, beta, upstream, eps=1e-5, show=False):
    custom = LayerNorm2D(x.shape[1], eps=eps, train=True, affine=True)
    custom.gamma.value = gamma.reshape(1, -1).copy()
    custom.beta.value = beta.reshape(1, -1).copy()
    reference = torch.nn.LayerNorm(x.shape[1], eps=eps, dtype=torch.float64)
    with torch.no_grad():
        reference.weight.copy_(torch.from_numpy(gamma))
        reference.bias.copy_(torch.from_numpy(beta))

    tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    expected_y = reference(tx)
    # With upstream == 1 this is exactly output.sum().backward().
    (expected_y * torch.from_numpy(upstream)).sum().backward()
    actual_y = custom.forward(x.copy())
    actual_dx = custom.backward(upstream.copy())
    expected_y = expected_y.detach().numpy()
    expected_dx = tx.grad.detach().numpy()
    np.testing.assert_allclose(actual_y, expected_y, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual_dx, expected_dx, rtol=1e-10, atol=1e-10)
    errors = (np.max(np.abs(actual_y - expected_y)),
              np.max(np.abs(actual_dx - expected_dx)))
    print(f"PASS {name}: max |output error|={errors[0]:.3e}, "
          f"max |input gradient error|={errors[1]:.3e}")
    if show:
        for label, value in [("input", x), ("custom output", actual_y),
                             ("torch output", expected_y),
                             ("custom input gradient", actual_dx),
                             ("torch input gradient", expected_dx)]:
            print(f"{label}:\n{value}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compat-var", action="store_true",
                        help="Diagnose math using an ndarray subclass that maps correction to ddof")
    args = parser.parse_args()
    np.set_printoptions(precision=10, suppress=True)
    print(f"NumPy {np.__version__}; PyTorch {torch.__version__}; CPU float64")
    if args.compat_var:
        print("DIAGNOSTIC MODE: ndarray.var correction is mapped to ddof; source is unchanged.")
    x = np.array([[1., 2., 4.], [-2., 0., 3.]])
    if args.compat_var:
        x = x.view(CompatibleArray)
    gamma = np.array([0.5, -1., 2.])
    beta = np.array([0.1, 0.2, -0.3])
    results = [compare("explicit output.sum()", x, gamma, beta,
                       np.ones_like(x), show=True)]
    rng = np.random.default_rng(42)
    for batch, features in [(1, 1), (1, 7), (4, 3), (8, 32)]:
        gamma = rng.normal(size=features)
        beta = rng.normal(size=features)
        inputs = {
            "random": rng.normal(size=(batch, features)),
            "constant rows": np.full((batch, features), 3.),
            "near constant rows": 3. + 1e-7 * rng.normal(size=(batch, features)),
        }
        for kind, x in inputs.items():
            if args.compat_var:
                x = x.view(CompatibleArray)
            for loss, upstream in [("sum", np.ones_like(x)),
                                   ("weighted sum", rng.normal(size=x.shape))]:
                results.append(compare(f"{x.shape} {kind}, {loss}",
                                       x, gamma, beta, upstream))
    print(f"All {len(results)} cases passed (rtol=atol=1e-10).")
    output_error, gradient_error = np.max(results, axis=0)
    print(f"Overall maximum errors: output={output_error:.3e}, "
          f"input gradient={gradient_error:.3e}")


if __name__ == "__main__":
    main()
