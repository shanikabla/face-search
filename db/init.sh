#!/bin/bash
# Runs only on the first start of the postgres container (empty data volume).
# Airflow's metadata lives in the DB created by POSTGRES_DB;
# our application data lives in a separate DB, owned by its own user.
set -euo pipefail

# psql -v + :'var' quotes the values safely (no SQL injection via the password)
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  -v faces_user="$FACES_DB_USER" \
  -v faces_password="$FACES_DB_PASSWORD" \
  -v faces_db="$FACES_DB_NAME" <<'SQL'
CREATE USER :"faces_user" WITH PASSWORD :'faces_password';
CREATE DATABASE :"faces_db" OWNER :"faces_user";
\connect :"faces_db"
CREATE EXTENSION IF NOT EXISTS vector;
SQL
