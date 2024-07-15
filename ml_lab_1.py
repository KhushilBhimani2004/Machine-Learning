import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load your dataset
@st.cache_data # Cache the dataframe for better performance
def load_data():
    df = pd.read_csv("C://Users//hp pc//Downloads//housing.csv")  # Adjust path as needed
    return df

# Data loading
df = load_data()

# Data preprocessing
df_clean = df.dropna(axis=0, how='any')  # Drop rows with any NaN values

# Encode categorical variable 'ocean_proximity'
label_encoder = LabelEncoder()
df_clean['ocean_proximity_encoded'] = label_encoder.fit_transform(df_clean['ocean_proximity'])

# Feature selection
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'ocean_proximity_encoded']

# Train the model
X = df_clean[features]
y = df_clean['median_house_value']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lm = LinearRegression()
lm.fit(X_scaled, y)

# Sidebar navigation
rad = st.sidebar.radio("Navigation",["Home", "Visualization", "Prediction Models", "About Us"])

if rad == "Home":
    st.title("Home: Dataset Overview")
    if st.checkbox("Show Dataset"):
        st.dataframe(df_clean)  # Display the cleaned dataset

elif rad == "Visualization":
    st.title("Data Visualization")
    st.write("Scatter plot of longitude vs latitude")
    fig, ax = plt.subplots()
    sns.scatterplot(x='longitude', y='latitude', data=df_clean.sample(3000), s=32, alpha=0.8, ax=ax)
    st.pyplot(fig)  # Display plot using st.pyplot()

    st.header("Correlation Heatmap")

    # Filter only numeric columns for correlation matrix
    numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    corr_matrix = df_clean[numeric_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, ax=ax)
    st.pyplot(fig)  # Display plot using st.pyplot()

    st.header("Histograms of Selected Features")
    selected_features = st.multiselect("Select features to plot histograms", df_clean.columns)
    if selected_features:
        for feature in selected_features:
            fig, ax = plt.subplots()
            sns.histplot(df_clean[feature], bins=50, ax=ax)
            ax.set_title(feature)
            st.pyplot(fig)  # Display each histogram using st.pyplot()

elif rad == "Prediction Models":
    st.title("Prediction Model: Housing Price Prediction")

    # User input for prediction
    st.header("Input Features for Prediction")
    input_features = {}
    for feature in features:
        if feature == 'ocean_proximity_encoded':
            input_features[feature] = st.selectbox(f"Select {feature}", df_clean['ocean_proximity'].unique())
        else:
            input_features[feature] = st.number_input(f"Enter {feature}", value=0.0)

    # Prediction button
    if st.button("Predict Price"):
        # Encode ocean_proximity for prediction
        input_features['ocean_proximity_encoded'] = label_encoder.transform([input_features['ocean_proximity_encoded']])[0]

        # Prepare input array for prediction
        input_array = np.array([input_features[feature] for feature in features]).reshape(1, -1)

        # Scale input features
        input_array_scaled = scaler.transform(input_array)

        # Predict housing price
        predicted_price = lm.predict(input_array_scaled)

        st.subheader("Predicted Housing Price")
        st.write(f"The predicted housing price is: ${predicted_price[0]:,.2f}")

elif rad == "About Us":
    st.title("About Us")
    st.header("Project Information")
    st.markdown("This Streamlit app demonstrates a simple integration of data loading, preprocessing, model training, and visualization for housing price prediction using linear regression.")
    st.header("Developed by")
    st.markdown("Khushil Girish Bhimani")
    st.markdown("[GitHub](https://github.com/KhushilBhimani2004/Machine-Learning)")

# Add any other sections or interactive elements you need
