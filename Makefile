

conda_create: 
	conda create -n predict_plays -c conda-forge numpy pandas jupyter matplotlib scikit-learn seaborn

conda: 
	conda info --envs

# https://www.anaconda.com/docs/tools/working-with-conda/environments#locking-an-environment
# conda env export > environment.yml

# https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084
# python -m ipykernel install --user --name=predict_plays