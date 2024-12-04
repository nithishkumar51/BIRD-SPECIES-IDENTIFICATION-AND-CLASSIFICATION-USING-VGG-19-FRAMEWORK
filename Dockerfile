FROM python:3.10.6

# set a directory for the app
WORKDIR /project/

# using the default streamlit port number for container to expose
EXPOSE 8501

# copy all the files to the container
COPY app.py /project/
COPY requirements.txt /project/

# install dependencies
RUN apt-get update
RUN pip install --no-cache-dir -r requirements.txt

# defining arguments
ARG A_U
ARG A_K
ARG S_K
ARG S_B

# defining environment variables
ENV API_URL $A_U
ENV ACCESS_KEY $A_K
ENV SECRET_KEY $S_K
ENV S3_BUCKET $S_B

# running the streamlit app
ENTRYPOINT ["streamlit", "run", "app.py"]