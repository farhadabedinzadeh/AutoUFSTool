# Auto-UFSTool - An Automatic MATLAB Toolbox for Unsupervised Feature Selection


## Abstract

- Several open resource toolboxes provide feature selection algorithms to decrease redundant features, data dimensionality, and computing costs.
These approaches demand programming expertise, limiting their popularity, and they haven't adequately addressed unlabeled real-world data. Automatic MATLAB Toolbox for Unsupervised Feature Selection (Auto-UFSTool) is a library for 23 robust Unsupervised Feature Selection techniques. Our goal is to develop a user-friendly and fully-automatic toolbox utilizing various unsupervised feature selection methodologies from latest research. It is freely available in MATLAB File Exchange repository and each technique's script and source code are included. Therefore, without requiring a single line of code, a clear and systematic comparison of alternative approaches is possible.

## Introduction
* This toolbox offers more than 20 Unsupervised Feature Selection methods.
* Almost half of them have been proposed in the last five years.
* This toolbox is user-friendly. After loading the data, users may launch certain procedures and applications without writing a single line of code.

## Usage
In the presence of an input matrix X(m×n)(m samples and n features per samples), the process for utilizing one of the UFS methods in the toolbox is as follows:
```code
Result = Auto_UFSTool(X,Selection_Method);    (1)
```
where Result represents the output rank indexes of features in descending order of their relative importance or subset of feature.
As illustrated in (1) a user can utilize any UFS method using an interface `main.m`.

* *`Result`*   : Rank indexes of features in descending order of their relative importance or Feature subset.
* *`Selection_Method`*  : Selected Unsupervised Feature Selection Method
* *`X(m×n)`*   : parameter settings
    + *`m`* : Samples
    + *`n`* : Features per samples



It is demonstrated with an example. Based on the `COIL20` dataset.The COIL20 is a library of images from Columbia containing 20 objects. As each object is rotated on a turntable, 72 images were captured at 5 degrees apart, and each object contains 72 images. Each image is 32 by 32 pixels and contains 256 grey levels per pixel.
As a result, with the input X, `m = 1440` and `n = 1024`.
After loading the data, one line of code to utilize the Unsupervised Feature Selection via Adaptive Graph Learning and Constraint (`EGCFS`) algorithm is presented below. 
```code
Result=Auto_UFSTool(X,'EGCFS')                (2)    
```

## Note
- It is important to note that all the options and parameters of the methods will be automatically received from the user or their default values may be used when the method is implemented, not to mention that all UFS methods' names are mentioned in the `UFS_Names.mat` file. For any further information, kindly see the original publications and algorithm implementations.
- The toolbox is written in MATLAB, a prominent programming language for machine learning and pattern recognition research.
The Auto-UFSTool was tested on 64bit Windows 8/10/11 PCs with MATLAB R2019b/R2022a on a range of publicly available datasets based on original articles
- To run this Code, you will need to add the `functions` and `UFSs`folder to your MATLAB path
And then run `main.m`.
- The Auto-UFSTool was tested on 64bit Windows 8/10/11 PCs with MATLAB R2019b/R2022a on a range of publicly available datasets based on original articles.


## Table1: UFS names, their Type which is f = filters, w = wrappers, h = hybrid, and e = embedding methods, the abbreviation of their names

| No.  | Abbreviation | Article Name                                                                                                                                 |
|------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | `'CFS'`      | [Gene Selection for Cancer Classification using Support Vector Machines](https://doi.org/10.1023/A:1012487302797)                            |
| 2    | `'LS'`       | [Laplacian Score for Feature Selection](https://www.researchgate.net/publication/221619142_Laplacian_Score_for_Feature_Selection)            |
| 3    | `'SPEC'`     | [Spectral Feature Selection for Supervised and Unsupervised Learning](https://doi.org/10.1145/1273496.1273641)                               |
| 4    | `'MCFS'`     | [Unsupervised feature selection for Multi-Cluster data](https://doi.org/10.1145/1835804.1835848)                                             |
| 5    | `'UDFS'`     | [ℓ2,1-Norm regularized discriminative feature selection for unsupervised learning](https://doi.org/10.5591/978-1-57735-516-8%2FIJCAI11-267)  |
| 6    | `'LLCFS'`    | [Feature Selection and Kernel Learning for Local Learning-Based Clustering](https://doi.org/10.1109/TPAMI.2010.215)                          |
| 7    | `'NDFS'`     | [Unsupervised Feature Selection Using Nonnegative Spectral Analysis](https://doi.org/10.1609/aaai.v26i1.8289)                                |
| 8    | `'RUFS'`     | [Robust Unsupervised Feature Selection](https://www.researchgate.net/publication/262217573_Robust_Unsupervised_Feature_Selection)            |
| 9    | `'FSASL'`    | [Unsupervised feature selection with adaptive structure learning](https://doi.org/10.1145/2783258.2783345)                                   |
| 10   | `'SCOFS'`    | [Unsupervised Simultaneous Orthogonal Basis Clustering Feature Selection](https://doi.org/10.1109/CVPR.2015.7299136)                         | 
| 11   | `'SOGFS'`    | [Unsupervised Feature Selection with Structured Graph Optimization](https://doi.org/10.1609/aaai.v30i1.10168)                                |
| 12   | `'UFSOL'`    | [Unsupervised feature selection with ordinal locality](https://doi.org/10.1109/ICME.2017.8019357)                                            |
| 13   | `'Inf-FS'`   | [Infinite Feature Selection](https://doi.org/10.1109/ICCV.2015.478)                                                                          |
| 14   | `'DGUFS'`    | [Dependence guided unsupervised feature selection](https://doi.org/10.1609/aaai.v32i1.11904)                                                 |
| 15   | `'SRCFS'`    | [Unsupervised feature selection with multi-subspace randomization and collaboration](https://doi.org/10.1016/j.knosys.2019.07.027)           | 
| 16   | `'CNAFS'`    | [Convex Non-Negative Matrix Factorization With Adaptive Graph for Unsupervised Feature Selection](https://doi.org/10.1109/tcyb.2020.3034462) | 
| 17   | `'EGCFS'`    | [Unsupervised Feature Selection via Adaptive Graph Learning and Constraint](https://doi.org/10.1109/TNNLS.2020.3042330)                      | 
| 18   | `'RNE'`      | [Robust neighborhood embedding for unsupervised feature selection](https://doi.org/10.1016/j.knosys.2019.105462)                             | 
| 19  | `'Inf-FS2020'`| [Infinite Feature Selection: A Graph-based Feature Filtering Approach](https://doi.org/10.1109/TPAMI.2020.3002843)                           | 
|20 | `'UAR-HKCMI'`| [Fuzzy complementary entropy using hybrid-kernel function and its unsupervised attribute reduction](https://doi.org/10.1016/j.knosys.2021.107398) 
|21 | `'FMIUFS'`   | [A Novel Unsupervised Approach to Heterogeneous Feature Selection Based on Fuzzy Mutual Information](https://doi.org/10.1109/TFUZZ.2021.3114734)| 
| 22   | `'FRUAR'`    | [Unsupervised attribute reduction for mixed data based on fuzzy](https://doi.org/10.1016/j.ins.2021.04.083)                                  | 
| 23   | `'U2FS'`     | [Utility metric for unsupervised feature selection](https://doi.org/10.7717/peerj-cs.477)                                                    |

## Documentation
For further questions, please see the appendix or feel free to contact developers.

- [Appendix](https://mega.nz/file/2UJXxJqJ#IXsiEcRe17WEH8oqvojUy4HLDAw39JzZaiR_Hw7wgNc)

[![Mail](https://img.shields.io/badge/Gmail-farhaad.abedinzadeh%40gmail.com-critical?style=flat-square&logo=gmail)]()
[![Yahoo - y.modaresnia@yahoo.com](https://img.shields.io/badge/Yahoo-y.modaresnia%40yahoo.com-2ea44f?logo=https%3A%2F%2Fmega.nz%2Ffile%2FbIoEnRRA%23CvFs356RFPPvv1BCUYhoAOotCI2xU8t2jgCkijOWKUs)](y.modaresnia@yahoo.com)
