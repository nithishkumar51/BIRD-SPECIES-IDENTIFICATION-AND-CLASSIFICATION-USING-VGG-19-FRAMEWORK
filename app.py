import numpy as np
import pandas as pd
import cv2 as cv2
from PIL import Image
import streamlit as st
import boto3
import requests
import os
from io import BytesIO
import socket

API_URL = os.environ.get('API_URL')
ACCESS_KEY = os.environ.get('ACCESS_KEY')
SECRET_KEY = os.environ.get('SECRET_KEY')
bucket_name = os.environ.get('S3_Bucket')
session = boto3.Session(
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)
s3 = session.resource('s3')
csv_key = 'data/birds_species.csv'

@st.cache_data()
def api_request(api_url: str, image: Image) -> str:
    response = requests.post(api_url, data = image, headers = {'Content-Type' : 'image/jpeg'})
    return response.json()

@st.cache_data()
def load_csv(csv_key: str, bucket_name: str):
    obj = s3.Bucket(bucket_name).Object(csv_key).get()['Body'].read()
    birds_df = pd.read_csv(BytesIO(obj))
    return birds_df

@st.cache_resource()
def load_files_from_s3(keys: list, bucket_name: str):
    images_lst = []
    for key in keys:
        obj = s3.Bucket(bucket_name).Object(key).get()['Body'].read()
        image = Image.open(BytesIO(obj))
        images_lst.append(image)
    return images_lst

if __name__ == '__main__':
    birds_df = load_csv(csv_key, bucket_name)
    test_df = birds_df[birds_df['Dataset'] == 'test'].reset_index()
    classes = birds_df['Species'].unique()
    height = 128
    width =128
    
    st.markdown("# Welcome To Bird Species Identification!:bird:")
    st.markdown("""Upload your image of a bird and the Species of the bird
            will be will be predicted by a Deep Neural Network in
            real-time and displayed on the screen. Top Five most
            likely Bird Species along with confidence level will
            also be displayed.
            """)
    st.info("Data obtained from the [Kaggle Dataset](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) by Gerald Piosenka.")
    
    file = st.file_uploader("Upload A Bird Image")
    if file:
        image = Image.open(file)
        img_file = BytesIO()
        image.save(img_file, format = 'JPEG')
        st.markdown("## Here is the Image You have uploaded.")
        st.image(image)
        y_pred_prob = np.array(api_request(API_URL, img_file.getvalue()))
        y_pred = np.argmax(y_pred_prob, axis = 1)[0]
        y_top5_prob = np.sort(y_pred_prob)[:, -1:-6:-1]
        y_top5_label = np.argsort(y_pred_prob)[:, -1:-6:-1]
        top5 = list(zip(y_top5_label[0], y_top5_prob[0]))
        
        st.success(f"The Bird belongs to **{classes[y_pred]}** Species.")
        
        df = pd.DataFrame(data = np.zeros((5, 2)), columns = ['Species', 'Confidence Level'],
              index = np.linspace(1, 5, 5, dtype = int), dtype = 'string')
        for i, (label, prob) in enumerate(top5):
            df.iloc[i, 0] = classes[label]
            df.iloc[i, 1] = str(np.round((prob * 100), 4)) + "%"
        
        st.markdown("## Here are the five most likely Bird Species.")
        st.dataframe(df)

        st.markdown(f"## Here are some other images of {classes[y_pred]}.")
        key_lst = []
        for j in range(3):
            path = test_df[test_df['Species'] == classes[y_pred]]['Filepath'].values[j]
            key_lst.append(path)
        image_lst = load_files_from_s3(key_lst, bucket_name)
        st.image(image_lst)
    st.info(f'Hostname: **{socket.gethostname()}**')