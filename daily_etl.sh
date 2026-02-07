#!/bin/bash
# Daily ETL Scheduler Script
# Run this in a cron job at 2 AM daily: 0 2 * * * /path/to/daily_etl.sh

set -e

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ETL_SCRIPT="$PROJECT_ROOT/src/etl/etl_main.py"
LOG_DIR="$PROJECT_ROOT/logs"
DOCKER_COMPOSE="$PROJECT_ROOT/docker/docker-compose.yml"
ALERT_EMAIL="${ALERT_EMAIL:-admin@example.com}"

# Create log directory
mkdir -p "$LOG_DIR"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_DIR/scheduler.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "========================================="
log "Daily ETL Pipeline Started"
log "========================================="

# Check if Docker is running
if ! docker ps > /dev/null 2>&1; then
    log "ERROR: Docker is not running"
    exit 1
fi

# Check if PostgreSQL container is running
if ! docker-compose -f "$DOCKER_COMPOSE" ps postgres | grep -q "Up"; then
    log "WARNING: PostgreSQL container not running. Starting..."
    docker-compose -f "$DOCKER_COMPOSE" up -d
    sleep 10
fi

# Check if there are files in landing directory
LANDING_DIR="$PROJECT_ROOT/data/landing"
FILE_COUNT=$(find "$LANDING_DIR" -type f -name "*.csv" 2>/dev/null | wc -l)

if [ "$FILE_COUNT" -eq 0 ]; then
    log "No CSV files in landing directory. Skipping ETL execution."
    exit 0
fi

log "Found $FILE_COUNT CSV file(s) in landing directory"

# Run ETL pipeline
log "Executing ETL pipeline..."

cd "$PROJECT_ROOT"

if python3 "$ETL_SCRIPT"; then
    log "SUCCESS: ETL pipeline completed successfully"
    STATUS="SUCCESS"
else
    log "ERROR: ETL pipeline failed"
    STATUS="FAILED"
fi

# Clean old logs (older than 30 days)
log "Cleaning logs older than 30 days..."
find "$LOG_DIR" -name "*.log" -type f -mtime +30 -delete
find "$LOG_DIR" -name "*.json" -type f -mtime +30 -delete

# Send email notification (if configured)
if [ ! -z "$ALERT_EMAIL" ] && [ "$STATUS" = "FAILED" ]; then
    log "Sending failure notification to $ALERT_EMAIL"
    # This would require mail/sendmail configured on the system
    # Example: echo "ETL failed" | mail -s "ETL Pipeline Alert" "$ALERT_EMAIL"
fi

log "========================================="
log "Daily ETL Pipeline Completed - Status: $STATUS"
log "========================================="
