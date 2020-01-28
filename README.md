# 02_NTU_workshop_Python_notebook
## Requirements
 ### 1. Operating system
 There is no special requirement for the system type. If your system supports Java and Python you should be able to run the case. The Jupyter notebook was thoroughly tested on linux based system (openSUSE Leap 15.0 and 15.1), as well as Windows based machines (Windows 10).
 ### 2. Software
 In order to run Jupyter notebook you should install __Java__ and __Python__.  
 According to h2o.ai website (http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html) Java 8, 9, 10, 11, 12, and 13 are supported. We will be running  binary package of H2O so JRE is required.  
 Python version 3.6.5 is required. It is advisable to create separate Python environment. Using Anaconda platform is recommended (https://www.anaconda.com/). Detailed step-by-step guide on installation of H2O and Python is available at: https://nbviewer.jupyter.org/github/jszlek/02_NTU_workshop_Python_notebook/blob/master/h2o_PLGA_case.ipynb  section `2. H2O installation` and `Installing Anaconda & h2o (Python)`.  
 If you wish to run the case on your machine please clone or download the repository (https://github.com/jszlek/02_NTU_workshop_Python_notebook/) and open it via Jupyter notebook.  
 To unpack the 7-zip files which are stored in the repository (`*.7z`) please install a software from https://www.7-zip.org/ or use your package manager (`zypper`, `apt-get`)
 ### 3. Hardware
 Please, be advised that running H2O server can cause your machine to _freeze_ or even _crash_, if it has not enough resources (RAM and/or CPUs). If you wish to use the notebook on your own data sets please carefully set the `my_max_ram_allowed` variable not to exceed your free RAM.

## Preview
To view the case please visit:
https://nbviewer.jupyter.org/github/jszlek/02_NTU_workshop_Python_notebook/blob/master/h2o_PLGA_case.ipynb  

## Batch mode
In order to run a script in a batch mode a Python script was prepared (https://github.com/jszlek/02_NTU_workshop_Python_notebook/blob/master/AutoML_Jupyter_notebook.py). Please edit the script according to your needs, save it, switch to the case directory, run terminal and activate `h2o` environment from your anaconda prompt. Then type `python AutoML_Jupyter_notebook.py`.   

If something goes wrong and you will have troubles in getting everything running, please do not hesitate to contact me: j.szlek[at]uj.edu.pl
