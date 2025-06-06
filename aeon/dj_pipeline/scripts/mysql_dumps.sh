#!/bin/bash

# Script to perform MySQL dumps of archived schemas
# Usage: ./mysql_dumps.sh [mysql_host] [mysql_user] [mysql_password]

set -e  # Exit on error

# Configuration
SCHEMA_PREFIX="aeon_archived_exp01_"  # Change this to match your schema prefix
MYSQL_HOST=${1:-"localhost"}  # Default to localhost if not provided
MYSQL_USER=${2:-"root"}      # Default to root if not provided
MYSQL_PASSWORD=${3:-"simple"} # Default password from docker config
DUMP_DIR="/ceph/aeon/aeon/dj_store/mysql_dumps/aeon_archived_exp01"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${DUMP_DIR}/mysql_dump_${TIMESTAMP}.log"

# Create dump directory if it doesn't exist
mkdir -p "${DUMP_DIR}"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# Function to perform dump for a single schema
dump_schema() {
    local schema=$1
    local dump_file="${DUMP_DIR}/${schema}_${TIMESTAMP}.sql"
    
    log_message "Starting dump of schema: ${schema}"
    
    if mysqldump -h"${MYSQL_HOST}" -u"${MYSQL_USER}" -p"${MYSQL_PASSWORD}" \
        --single-transaction \
        --routines \
        --triggers \
        --events \
        --add-drop-database \
        --databases "${schema}" > "${dump_file}" 2>> "${LOG_FILE}"; then
        
        log_message "Successfully dumped schema: ${schema}"
        # Compress the dump file
        gzip "${dump_file}"
        log_message "Compressed dump file: ${dump_file}.gz"
    else
        log_message "ERROR: Failed to dump schema: ${schema}"
        return 1
    fi
}

# Main execution
log_message "Starting MySQL dump process"

# Get list of schemas with specified prefix
SCHEMAS=$(mysql -h"${MYSQL_HOST}" -u"${MYSQL_USER}" -p"${MYSQL_PASSWORD}" -N -e "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME LIKE '${SCHEMA_PREFIX}%';")

if [ -z "${SCHEMAS}" ]; then
    log_message "No schemas found with prefix: ${SCHEMA_PREFIX}"
    exit 0
fi

# Print configuration and schemas to be dumped
echo "MySQL Dump Configuration"
echo "----------------------"
echo "Host: ${MYSQL_HOST}"
echo "User: ${MYSQL_USER}"
echo "Schema prefix: ${SCHEMA_PREFIX}"
echo "Dump directory: ${DUMP_DIR}"
echo ""
echo "The following schemas will be dumped:"
echo "------------------------------------"
for schema in ${SCHEMAS}; do
    echo "- ${schema}"
done 
echo ""

# Ask for confirmation
read -p "Do you want to proceed with the dump? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Dump cancelled by user"
    exit 1
fi

# Perform dumps
for schema in ${SCHEMAS}; do
    dump_schema "${schema}"
done

log_message "MySQL dump process completed"

# Print summary
echo "Dump Summary:"
echo "-------------"
echo "Schema prefix: ${SCHEMA_PREFIX}"
echo "MySQL host: ${MYSQL_HOST}"
echo "Dump directory: ${DUMP_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Number of schemas dumped: $(echo "${SCHEMAS}" | wc -l)"
echo "Timestamp: ${TIMESTAMP}" 