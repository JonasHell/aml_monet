We are something of a painter ourselves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Team: Daniel Galperin, Jonas Hellgoth, Alexander Kunkel

About
^^^^^^^^^^^^^^^^

Using conditional invertible neural networks for image-to-image translation with landscape photos and Monet paintings.
Our source code heavily draws on https://github.com/VLL-HD/conditional_INNs and the corresponding publication https://arxiv.org/abs/1907.02392.

Directories
^^^^^^^^^^^^^^^^
+---------------------------+--------------------------------------------------+
| config.py                 | hyperparameters and paths                        |
+---------------------------+--------------------------------------------------+
| data.py                   | load data from path specified in config.py       |
+---------------------------+--------------------------------------------------+
| models.py                 | includes final architecture MonetCINN_squeeze    |
+---------------------------+--------------------------------------------------+
| train.py & eval.py        | train and evaluate models                        |
+---------------------------+--------------------------------------------------+


The model
^^^^^^^^^^^^^^^^
The trained model used to generate all figures in the report can be downloaded here:
https://drive.google.com/file/d/1obP2slgHca-HhP31gpaQIm5Qs374-4pT/view?usp=sharing

For loading, set appropriate paths in config.py.

Dependencies
^^^^^^^^^^^^^^^^

All version numbers are only the minimum version required to run the code. Probably, most other versions will work too. 

+---------------------------+-------------------------------+
| **Package**               | **Version**                   |
+---------------------------+-------------------------------+
| Pytorch                   | 1.8.0                         |
+---------------------------+-------------------------------+
| Numpy                     | 1.19.5                        |
+---------------------------+-------------------------------+
| Matplotlib                | 3.2.2                         |
+---------------------------+-------------------------------+
| scikit-learn              | 0.22.2                        |
+---------------------------+-------------------------------+
| PIL                       | 0.1.12                        |
+---------------------------+-------------------------------+
| albumentations            | 7.1.2                         |
+---------------------------+-------------------------------+
| cv2                       | 4.1.2                         |
+---------------------------+-------------------------------+
