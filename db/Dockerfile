FROM postgres:latest

# Set environment variables
ENV POSTGRES_DB chat
ENV POSTGRES_USER test
ENV POSTGRES_PASSWORD test

# Init db scripts folder and assign to postgres user
#RUN mkdir /docker-entrypoint-initdb.d/
#RUN chown postgres:postgres /docker-entrypoint-initdb.d

#COPY data/* /docker-entrypoint-initdb.d/
#COPY data/init.sql /docker-entrypoint-initdb.d/init.sql
#RUN chown postgres:postgres /docker-entrypoint-initdb.d/*

RUN usermod -aG sudo postgres

# Create a directory to store CSV files and copy them into the container
#COPY data /docker-entrypoint-initdb.d/data
USER postgres
RUN whoami
ENTRYPOINT ["docker-entrypoint.sh"]
EXPOSE 5432
CMD ["postgres"]
