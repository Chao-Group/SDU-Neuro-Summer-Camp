conda activate brats python=3.10 -y
pip install jupyter notebook ipykernel

# Add the environment to Jupyter
python -m ipykernel install --user --name=myenv --display-name="Python (brats)"


