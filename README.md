
Steerable PDO support for the E2CNN library
--------------------------------------------------------------------------------
**[Experiments](https://github.com/ejnnr/steerable_pdo_experiments)** | **[Paper](https://arxiv.org/abs/2106.10163)** | **[Original library](https://github.com/QUVA-Lab/e2cnn)**

This is a fork of the [*e2cnn* library](https://github.com/QUVA-Lab/e2cnn) that adds support
for [steerable PDOs](https://arxiv.org/abs/2106.10163), i.e. equivariant partial differential operators.

The main changes are a new `diffops` module that plays the analogous role to `kernels`
but for PDOs, a `gspace.build_diffop_basis()` method (analogous to `gspace.build_kernel_basis()`)
and the equivariant `nn.R2Diffop` module (a drop-in replacement for `nn.R2Conv`).

If you have questions specifically about Steerable PDOs and this implementation,
please [contact me](mailto:erik.jenner99@gmail.com).

## Installation
The original *e2cnn* library can be installed with `pip install e2cnn`.
If you want to install the main branch of this fork, use
```
pip install git+https://github.com/ejnnr/steerable_pdos
```
Note that if you want to use Gaussian or RBF-FD discretization, you will need the
[RBF library](https://github.com/treverhines/RBF) as an additional optional dependency.

If you want to reproduce our experiments, then do *not* install the library this
way, instead see our [experiments repository](https://github.com/ejnnr/steerable_pdo_experiments) for instructions.

## Cite

The original *e2cnn* library was developed as part of the paper
[General E(2)-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251). 
Please cite this work if you use the library:

```
@inproceedings{e2cnn,
    title={{General E(2)-Equivariant Steerable CNNs}},
    author={Weiler, Maurice and Cesa, Gabriele},
    booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
    year={2019},
}
```

For the implementation of steerable PDOs inside this library, please cite [our paper](https://arxiv.org/abs/2106.10163):

```
@misc{jenner2021steerable,
    title={Steerable Partial Differential Operators for Equivariant Neural Networks}, 
    author={Erik Jenner and Maurice Weiler},
    year={2021},
    eprint={2106.10163},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


## License

*e2cnn* is distributed under BSD Clear license. See LICENSE file.
