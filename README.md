
Steerable PDO support for the E2CNN library
--------------------------------------------------------------------------------
This is a fork of the [*e2cnn* library](https://github.com/QUVA-Lab/e2cnn) that adds support
for Steerable PDOs, i.e. equivariant partial differential operators.

This branch (`experiments`) contains the original code for [our experiments](https://github.com/ejnnr/steerable_pdo_experiments).
Unless you want to reproduce these experiments, we strongly recommend you use
the [main branch](https://github.com/ejnnr/steerable_pdos) instead.

The differences to the main branch are:
- Slight differences in the library interface (the main branch tries harder to be
  backward compatible to e2cnn)
- This branch supports the basis used by [PDO-eConvs](https://arxiv.org/abs/2007.10408).
  If you are interested in using the PDO-eConv basis for your own work, feel free to
  file an issue or [contact me](mailto:erik.jenner99@gmail.com) (this part of the implementation
  was only ever designed for our experiments and is not as user-friendly as the remaining library).
- This branch has **experimental** support for convolutions and differential operators
  on point clouds (via `nn.R2GeneralConv` and `nn.R2GeneralDiffop`). Again, feel free to contact
  me if you are interested in using that; the implementation is relatively complete but
  rather slow and memory-intensive, and not as well documented.
- This branch is missing some features that we didn't need for the experiments but implemented
  for completeness sake in the main branch (e.g. support for O(2)).
- The main branch has complete and accurate documentation, whereas parts of this branch are
  essentially undocumented.

## Installation
If you want to set up this library for our experiments, see the instructions in
[our experiments repo](https://github.com/ejnnr/steerable_pdo_experiments).
If you want to use it in your own code, you are probably in the wrong branch,
see the [main branch](https://github.com/ejnnr/steerable_pdos).

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

Our paper on Steerable PDOs will be published soon.

## License

*e2cnn* is distributed under BSD Clear license. See LICENSE file.
