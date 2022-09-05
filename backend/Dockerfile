FROM python:3.9.4-slim

# set work directory
WORKDIR /backend

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

EXPOSE 8000:8000


# install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy project
COPY . .

# CMD [ "echo" ]

CMD [ "python3", "predict.py" ]
# CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "80"]

# FROM ubuntu:20.04

# RUN apt-get install sl

# ENV PORT 5000

# CMD ["echo", "Bob"]

# # Create app directory
# RUN mkdir -p /usr/src/app
# WORKDIR /usr/src/app

# COPY . /usr/src/app

# RUN sudo apt-get update -y

# RUN sudo apt-get install -y python

# COPY /backend/requirements.txt ./

# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# COPY . .

# CMD [ "python3", "./backend/predict.py" ]
