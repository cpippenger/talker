# Base image
#FROM ubuntu:20.04
FROM postgres:latest

ENV POSTGRES_PASSWORD docker
ENV POSTGRES_DB world

# Required software
#RUN apt-get update && apt-get install -y software-properties-common
#RUN apt-get update
#RUN apt-get install  -y sudo

# Set container working directory
WORKDIR /db

# Copy api files
COPY ./db/* /db/

# Install requirements
#RUN systemctl enable postgresql.service
COPY /db/createdatabase.sql /docker-entrypoint-initdb.d/
#RUN service postgresql start; sudo -u postgres psql < /db/createdatabase.sql
#RUN echo "listen_addresses = '*'" >> /etc/postgresql/12/main/postgresql.conf 
#RUN echo "host    all     	all     0.0.0.0/0       md5" >> /etc/postgresql/12/main/pg_hba.conf

# Open ports
EXPOSE 5432

# Entry point to start unvicorn server
#CMD service postgresql start

