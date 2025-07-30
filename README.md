# cis9660-project-2

Deployed app URL: https://bennettjg-cis9660-project-2-app-le7thu.streamlit.app/.

To run the Streamlit app locally, clone the repository, navigate to the folder in a terminal, and use the command `streamlit run app.py`. Requirements are listed in `requirements.txt`.

Data used is a subset of the 2025-06-12 Kickstarter dataset (CSV format) available on https://webrobots.io/kickstarter-datasets/. Due to the large size of the dataset and hardware limitations on my personal computer, I chose to focus on data from 2024 and 2025. The code used to preprocess the data can be found in utils/process_data.py, and the processed dataset can be found in data/Full_Kickstarter_data.csv.

Model fitting and evaluation (including classification reports and visualizations) are available in the`Kickstarter_Campaign_Classification.ipynb` notebook. This notebook also makes use of the `yellowbrick` library for the K-means elbow visualization plot, which is not one of the Streamlit app's requirements since that model performed poorly.