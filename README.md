# dihedral.jl
Julia code: Add on to RadiiPolynomial.jl for $D_3,D_4,$ and $D_6$-symmetric Fourier sequences.
# Computer assisted proofs with symmetry, reduced set of Fourier indices



Table of contents:


* [Introduction](#introduction)
* [Laplacian Operator](#laplacian)
* [Utilisation and References](#utilisation-and-references)
* [License and Citation](#license-and-citation)
* [Contact](#contact)



# Introduction

This Julia code is an add to the package [RadiiPolynomial](https://github.com/OlivierHnt/RadiiPolynomial.jl). It constructs the mathematical objects (spaces, sequences, operators,...) usually found in the aforementioned package for $D_3,D_4,$ and $D_6$-symmetric functions. More specifically, it performs computations on a reduced set of Fourier coefficients. The reduced set depends on the symmetry chosen.

The computations can be made rigorous by using the package [IntervalArithmetic](https://github.com/JuliaIntervals/IntervalArithmetic.jl).


# Laplacian Operator

In [RadiiPolynomial](https://github.com/OlivierHnt/RadiiPolynomial.jl), derivative operators are provided. When using dihedral-symmetric sequences, the action of differentiation does not necessarily yield another dihedral-symmetric sequence. For simplicity, we provide the Laplacian operator as it's action on a dihedral-symmetric sequence is still a dihedral-symmetric sequence. More specifically,

$$Laplacian(2) = \frac{\partial^2}{\partial x_1^2} + \frac{\partial^2}{\partial x_2^2}$$
 
 # Utilisation and References
 
 The code is build using the following packages :
 - [RadiiPolynomial](https://github.com/OlivierHnt/RadiiPolynomial.jl) 
 - [IntervalArithmetic](https://github.com/JuliaIntervals/IntervalArithmetic.jl)
 
 By installing [RadiiPolynomial](https://github.com/OlivierHnt/RadiiPolynomial.jl), you automatically install [IntervalArithmetic](https://github.com/JuliaIntervals/IntervalArithmetic.jl). Hence, by installing the former, you can use the dihedral Fourier sequence structure.

 More details on using the operations can be found in the documentation for [RadiiPolynomial](https://github.com/OlivierHnt/RadiiPolynomial.jl).
 
 # License and Citation
 
  This code is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).
  
If you wish to use this code in your publication, research, teaching, or other activities, please cite it using the following BibTeX template:

```
@software{dihedral.jl,
  author = {Dominic Blanco},
  title  = {dihedral.jl},
  url    = {https://github.com/dominicblanco/dihedral.jl},
  note = {\url{https://github.com/dominicblanco/dihedral.jl},
  year   = {2026},
  doi = {10.5281/zenodo.18626129}
}
```
DOI : [10.5281/zenodo.18626129](https://zenodo.org/records/18626129) 


# Contact

You can contact me at :
dominic.blanco@mail.mcgill.ca
