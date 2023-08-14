-- Database: talker
\set ECHO all
-- DROP DATABASE IF EXISTS talker;

CREATE DATABASE talker;
\c talker 

ALTER USER postgres PASSWORD 'xxxPASSWORDxxx';
CREATE TABLE IF NOT EXISTS public.comment
(
    id uuid NOT NULL,
    commentor character(25) COLLATE pg_catalog."default" NOT NULL,
    comment text COLLATE pg_catalog."default",
    "time" timestamp without time zone NOT NULL,
    positive_score decimal,
    negative_score decimal,
    sentiment text COLLATE pg_catalog."default",
    CONSTRAINT comments_pkey PRIMARY KEY (id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.comment
    OWNER to postgres;

select * from comment;