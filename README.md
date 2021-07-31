
Steerable PDO support for the E2CNN library
--------------------------------------------------------------------------------
This is a fork of the [*e2cnn* library](https://github.com/QUVA-Lab/e2cnn) that adds support
for Steerable PDOs, i.e. equivariant partial differential operators.

This branch (`pdo_econvs`) contains support for reproducing
[PDO-eConvs](https://arxiv.org/abs/2007.10408) exactly within the e2cnn library.
Unless you need these features, we strongly recommend you use
the [main branch](https://github.com/ejnnr/steerable_pdos) instead.

If you do want to use the PDO-eConv basis for your own work, feel free to
file an issue or [contact me](mailto:erik.jenner99@gmail.com) if you run into
trouble or have questions (this part of the implementation
was only ever designed for our experiments and is not as user-friendly as the remaining library).

## Installation
The original *e2cnn* library can be installed with `pip install e2cnn`.
If you want to install this `pdo_econv` branch of this fork, use
```
pip install git+https://github.com/ejnnr/steerable_pdos@pdo_econv
```
Note that if you want to use Gaussian or RBF-FD discretization, you will need the
[RBF library](https://github.com/treverhines/RBF) as an additional optional dependency.

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