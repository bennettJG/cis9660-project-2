import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime
import json
import pickle
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

random_state = 1701

st.set_page_config(page_title="Kickstarter Success Prediction", layout="wide", page_icon="📦")

st.header('Kickstarter data explorer and project success predictor')
st.markdown(':warning: :red[Disclaimer: this is a student project for educational purposes only.] :warning:')

@st.cache_data
def load_data():
    KS_data = pd.read_csv("Full_Kickstarter_data.csv")
    KS_data.fillna({'creator_prev_projects':0, 'backing_action_count':0}, inplace=True)
    KS_data['launched_at'] = pd.to_datetime(KS_data['launched_at'])
    return(KS_data)
    
@st.cache_data
def process_data(KS_data):
    KS_data = KS_data[['state_num', 'usd_goal', 'parent_category', 'sub_category', 'backing_action_count', 'has_video', 'creator_prev_projects', 'launch_day', 'hours_from_noon', 'launch_month', 'country', 'prelaunch_activated', 'blurb_length', 'staff_pick', 'days_active']]
    X = KS_data.drop('state_num', axis=1)
    y = KS_data['state_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
    numeric_columns = ['blurb_length',  'hours_from_noon', 'creator_prev_projects', 'usd_goal', 'backing_action_count', 'days_active']
    categorical_columns = list(set(KS_data.columns) - set(numeric_columns) - set(['state_num']))
    ct = ColumnTransformer([("text_preprocess", OneHotEncoder(drop='first', sparse_output=False, max_categories=30), categorical_columns),
    ("num_preprocess", RobustScaler(), numeric_columns)],
    verbose_feature_names_out = False
    )
    ct.set_output(transform='pandas')
    pipe = Pipeline(steps=[('onehot', ct)])
    X_train_scaled = pipe.fit_transform(X_train)
    X_test_scaled = pipe.transform(X_test)
    # Return pipeline along with training and test sets so that
    # it can be used on user-input data.
    return pipe, X_train_scaled, X_test_scaled, y_train, y_test
    
@st.cache_resource
def fit_model(X_train, y_train):
    model = SVC(random_state=random_state, class_weight='balanced')
    model.fit(X_train, y_train)
    pickle.dump(model, open('Kickstarter SVM.sav', 'wb'))
    return model
    
@st.cache_data
def model_test(_model, X_test):
    y_pred = model.predict(X_test)
    return y_pred
    
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:2rem;
    padding-right: 20px;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

subcategory_dict = {"All categories": [],
"Film & Video": ['Narrative Film', 'Horror', 'Shorts', 'Thrillers', 'Drama', 'Comedy', 'Television', 'Documentary', 'Webseries', 'Animation' 'Science Fiction', 'Action', 'Experimental'],
"Fashion": ['Footwear', 'Apparel', 'Accessories', 'Jewelry', 'Childrenswear', 'Ready-to-wear'],
"Art": ['Sculpture', 'Performance Art', 'Public Art', 'Illustration','Textiles', 'Painting', 'Social Practice', 'Digital Art', 'Conceptual Art'],
"Crafts": ['Crochet', 'Embroidery', 'DIY'],
"Music": ['Faith', 'Rock', 'Country & Folk', 'Electronic Music', 'Indie Rock', 'Jazz', 'World Music', 'Metal', 'Hip-Hop', 'Classical Music', 'Comedy'],
"Publishing": ['Zines', 'Periodicals', 'Fiction', "Children's Books",  'Art Books', 'Poetry', 'Nonfiction', 'Academic', 'Anthologies' 'Young Adult'],
"Design": ['Interactive Design', 'Product Design', 'Toys', 'Architecture','Graphic Design', 'Civic Design'],
"Food": ['Food Trucks', 'Farms', 'Restaurants', 'Cookbooks', 'Small Batch',"Farmer's Markets", 'Drinks'],
"Games": ['Playing Cards', 'Tabletop Games', 'Video Games', 'Live Games','Mobile Games', 'Puzzles'], 
"Technology": ['3D Printing', 'Hardware', 'Gadgets', 'Apps', 'DIY Electronics', 'Web', 'Robots', 'Makerspaces', 'Wearables', 'Camera Equipment'],
"Photography": ['Photobooks', 'People', 'Places', 'Animals', 'Fine Art'],"Comics": ['Comic Books', 'Anthologies', 'Events', 'Graphic Novels'],
"Theater": ['Spaces', 'Experimental', 'Comedy', 'Plays', 'Musical','Immersive', 'Festivals'],
"Journalism": ['Web'],
"Dance": ['Performances', 'Residencies']}

KS_data = load_data()
pipe, X_train, X_test, y_train, y_test = process_data(KS_data)
model = pickle.load(open('Kickstarter SVM.sav', 'rb')) #fit_model(X_train, y_train) 
y_pred = model_test(model, X_test)

tab1, tab2, tab3 = st.tabs(["Predict campaign outcome", "About the model", "Explore the data"])

with tab1:
    st.header('Input project details:')
    usd_goal = st.number_input("Funding goal (US dollars):", min_value=0, value=1000, max_value=99999, width=200)
    col1, col2, col3 = st.columns(3)
    with col1:
        country_options = sorted(KS_data["country"].unique())
        country = st.selectbox("Your country:", country_options, index=24)
        launch_date = st.date_input("Planned campaign launch date:", min_value="today")
        launch_hour = st.time_input("Planned campaign launch time:", value='12:00', step=3600)
        days_active = st.number_input("Planned campaign length (days):", min_value=0, value=30, max_value=100)
    with col2:
        category = st.selectbox("Project category:", {k: v for k, v in subcategory_dict.items() if k != 'All categories'}, index=8)
        subcategory = st.selectbox("Subcategory:", subcategory_dict[category], placeholder="Select a category to see options", index=1)
        blurb = st.text_input("Blurb:", max_chars=160)
        creator_prev_projects = st.number_input("How many previous Kickstarter campaigns have you run?", min_value=0)
        backing_action_count = st.number_input("How many have you backed?", min_value=0)
    with col3:
        st.markdown("Does your project...")
        has_video = st.checkbox("Have a video?")
        prelaunch_activated = st.checkbox("Have prelaunch activated?")
        staff_pick = st.checkbox("Have a good chance of becoming a staff pick?")
    project_data = pd.DataFrame({'usd_goal': usd_goal, 'country':country, 'parent_category':category, 'sub_category':subcategory, 'backing_action_count':backing_action_count, 'has_video':has_video, 'creator_prev_projects':creator_prev_projects, 'launch_day':launch_date.strftime("%A"), 'hours_from_noon':abs(12-int(launch_hour.strftime("%H"))), 'launch_month':int(launch_date.strftime("%m")), 'prelaunch_activated':prelaunch_activated, 'blurb_length':len(blurb), 'staff_pick':staff_pick, 'days_active':days_active}, index=[0])
    prediction = ":green[Likely to reach goal]" if model.predict(pipe.transform(project_data))==[1] else ":red[Unlikely to reach goal]"
    st.markdown(f"## Model Prediction: {prediction}")
with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        plt.rc('font', size=6)
        st.markdown("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(2, 2))
        fig.patch.set_facecolor('#FAFAFA')
        st.pyplot(ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Failed','Successful'], ax=ax).figure_, use_container_width=False)
    with col2:
        st.markdown("### Classification Report")
        class_rpt = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        class_rpt.index = ['Failed', 'Successful', 'Accuracy', 'Macro Avg', 'Weighted Avg']
        st.dataframe(class_rpt)
    with col3:
        st.markdown("### ROC Curve")
        fig, ax = plt.subplots(figsize=(2,2))
        fig.patch.set_facecolor('#FAFAFA')
        st.pyplot(RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax).figure_, use_container_width=False)
with tab3:
    filtered_df = KS_data
    # Dataset filtering for exploration
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("## Dataset filters")
    with col2:
        ## Category
        category_filter = st.selectbox("Category", subcategory_dict.keys())
        subcategory_filter = st.selectbox("Subcategory", ['All subcategories'] + subcategory_dict[category_filter])
    with col3:    
        ## Funding goal
        min_goal = int(KS_data["usd_goal"].min())
        max_goal = int(KS_data["usd_goal"].max())

        goal_range = st.slider(
            "Funding Goal (US Dollars)",
            min_value=min_goal,
            max_value=max_goal,
            value=(min_goal, max_goal)
        )  
        ## Launch date
        min_date = filtered_df["launched_at"].dt.date.min()
        max_date = filtered_df["launched_at"].dt.date.max()
        start_date, end_date = st.date_input(
            "Launch Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date,
        )        
    with col4:
        ## Country
        country_options = ["All countries"]+sorted(KS_data["country"].unique())
        selected_country_filter = st.selectbox("Country", country_options)
    
    st.markdown("---")
    filtered_df = filtered_df[
        (filtered_df["launched_at"].dt.date >= start_date) &
        (filtered_df["launched_at"].dt.date <= end_date) &
        (filtered_df["usd_goal"] >= goal_range[0]) &
        (filtered_df["usd_goal"] <= goal_range[1]) 
    ]

    filtered_df = (filtered_df if selected_country_filter == "All countries" else 
        filtered_df[filtered_df["country"] == selected_country_filter])

    if (category_filter != "All categories"):
        filtered_df = filtered_df[filtered_df['parent_category'] == category_filter]
    if (subcategory_filter != "All subcategories"):
        filtered_df = filtered_df[filtered_df['sub_category'] == subcategory_filter]

    successful_df = filtered_df[filtered_df["state_binary"]=="Reached goal"]
    failed_df = filtered_df[filtered_df["state_binary"]=="Did not reach goal"]

    successfail_colors = {"Successful":'#05CE78', "Failed":'#C90B5D'}
    ## Data Explorer
    st.markdown("## Metrics")

    if len(filtered_df) == 0:
        st.markdown("# No projects matching specified filters!")
    else:
        st.markdown("### Projects")
        col1, col2, col3 = st.columns(3)
        with col1:
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("Total Projects", f'{len(filtered_df)}')
                success_fail = filtered_df["state_binary"].value_counts(sort = False).reset_index()
                fig = px.pie(success_fail,
                    values = "count",
                    names = "state_binary",
                    color="state_binary",
                    color_discrete_map = successfail_colors,
                    category_orders={'state_binary': ["Successful","Failed"]},
                    title = "Campaign Outcome")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            with col1b:
                st.metric("Unique Creators", f'{len(filtered_df[filtered_df['creator_prev_projects']==0])}')
        with col3:
            if (selected_country_filter == "All countries"):
                top_countries = (filtered_df['country'].value_counts().sort_values(ascending=False).head(10).
                    sort_values())
                top_countries_status = filtered_df[filtered_df['country'].isin(top_countries.index.to_list())]
                fig = px.bar(
                top_countries_status,
                    y="country",
                    color="state_binary",
                    title="Top 10 Countries by Project Count",
                    orientation="h",
                    category_orders={'state_binary': ["Successful","Failed"]},
                    color_discrete_map = successfail_colors,
                )
                fig.update_layout(
                    height=500,
                    yaxis_title="Country",
                    xaxis_title="# Projects",
                    barmode='stack', yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("# Remove the Country filtering to see top countries by project count!")
        with col2:
            if category_filter == "All categories":
                fig = px.bar(
                    filtered_df,
                    y="parent_category",
                    color="state_binary",
                    color_discrete_map = successfail_colors,
                    title="Project Categories",
                    category_orders={'state_binary': ["Successful","Failed"]},
                    orientation="h"
                )
                fig.update_layout(
                    height=500,
                    yaxis_title="Category",
                    xaxis_title="# Projects",
                    barmode='stack', yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
            elif subcategory_filter == "All subcategories":
                fig = px.bar(
                    filtered_df,
                    y="sub_category",
                    color="state_binary",
                    color_discrete_map = successfail_colors,
                    title="Project Categories",
                    orientation="h",
                    category_orders={'state_binary': ["Successful","Failed"]}
                )
                fig.update_layout(
                    height=500,
                    yaxis_title="Category",
                    xaxis_title="# Projects",
                    barmode='stack', yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("## Remove the Subcategory filtering to see project counts by category or subcategory!")
        st.markdown("---")
        st.markdown("### Funding")
        st.markdown("These plots frequently contain outliers - zoom in to see the distribution more clearly!")
        st.markdown("For ease of viewing, the top 1% of values are rounded down.")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(filtered_df, 
                x="usd_goal", 
                color = "state_binary",
                color_discrete_map = successfail_colors,
                title="Funding Goal Distribution",
                orientation = "h",
                category_orders={'state_binary': ["Successful","Failed"]},
                range_x = (0, filtered_df["usd_goal"].quantile(0.99))
                )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            stats_df = pd.DataFrame({
                    "Metric": ["Mean", "Median", "Std. Deviation", "Max", "Min"],
                    "Funding Goal (USD)": [
                        f"${filtered_df['usd_goal'].mean():,.2f}",
                        f"${filtered_df['usd_goal'].median():,.2f}",
                        f"${filtered_df['usd_goal'].std():,.2f}",
                        f"${filtered_df['usd_goal'].max():,.2f}",
                        f"${filtered_df['usd_goal'].min():,.2f}"
                    ]
                })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)  
            
        with col2:
            fig = px.box(filtered_df, 
                x="days_active", 
                color = "state_binary",
                color_discrete_map = successfail_colors,
                title="Campaign Length (Days)",
                category_orders={'state_binary': ["Successful","Failed"]},
                orientation = "h",
                range_x = (0, filtered_df["days_active"].quantile(0.99))
                )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            stats_df = pd.DataFrame({
                    "Metric": ["Mean", "Median", "Std. Deviation", "Max", "Min"],
                    "Campaign Length": [
                        f"${filtered_df['days_active'].mean():,.2f}",
                        f"${filtered_df['days_active'].median():,.2f}",
                        f"${filtered_df['days_active'].std():,.2f}",
                        f"${filtered_df['days_active'].max():,.2f}",
                        f"${filtered_df['days_active'].min():,.2f}"
                    ]
                })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)  

        st.markdown("---")
        st.markdown("## Trends over time")
        filtered_df['month_year'] = filtered_df["launched_at"].dt.strftime('%Y-%m')
        daily_launch = filtered_df[['month_year','state_binary']].value_counts().reset_index().sort_values('month_year')
        daily_launch.columns = ["Date", "Outcome", "Projects"]
        fig = px.line(daily_launch, x="Date", y="Projects", 
                    color='Outcome',
                    color_discrete_map = successfail_colors,
                    title="Monthly Projects Launched")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)