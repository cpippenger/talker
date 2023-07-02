-- Database: talker
\set ECHO all
-- DROP DATABASE IF EXISTS talker;

CREATE DATABASE talker;
\c talker 

ALTER USER postgres PASSWORD 'xxxPASSWORDxxx';
CREATE TABLE IF NOT EXISTS public."Comment"
(
    id uuid NOT NULL,
    commentor character(25) COLLATE pg_catalog."default" NOT NULL,
    comment text COLLATE pg_catalog."default",
    "time" timestamp without time zone NOT NULL,
    sentiment text COLLATE pg_catalog."default",
    CONSTRAINT comments_pkey PRIMARY KEY (id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public."Comment"
    OWNER to postgres;

select * from "Comment";