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
NOTIFY_EMAIL="${NOTIFY_EMAIL:-}"
FROM_EMAIL="${FROM_EMAIL:-ETL-notify@local}"
SUBJECT_PREFIX="${SUBJECT_PREFIX:-ETL Launch}"
HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Create log directory
mkdir -p "$LOG_DIR"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_DIR/scheduler.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

send_launch_email() {
    if [ -z "$NOTIFY_EMAIL" ]; then
        return 0
    fi

    subject="$SUBJECT_PREFIX: $HOSTNAME_SHORT"
    body="ETL pipeline launch detected on $HOSTNAME_SHORT at $(date '+%Y-%m-%d %H:%M:%S')."

    # Try Python email script first
    if [ -f "$PROJECT_ROOT/send_email.py" ] && [ -n "$PYTHON_BIN" ]; then
        if "$PYTHON_BIN" "$PROJECT_ROOT/send_email.py" "$NOTIFY_EMAIL" "$subject" "$body" 2>&1 | tee -a "$LOG_DIR/scheduler.log"; then
            log "Email notification sent to $NOTIFY_EMAIL"
            return 0
        fi
    fi

    # Fallback to mail/sendmail
    if command -v mail >/dev/null 2>&1; then
        echo "$body" | mail -s "$subject" -r "$FROM_EMAIL" "$NOTIFY_EMAIL"
        log "Email sent via mail command"
        return 0
    fi

    if command -v sendmail >/dev/null 2>&1; then
        {
            echo "From: $FROM_EMAIL"
            echo "To: $NOTIFY_EMAIL"
            echo "Subject: $subject"
            echo
            echo "$body"
        } | sendmail -t
        log "Email sent via sendmail"
        return 0
    fi

    log "WARNING: No email method available. See logs for details."
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
send_launch_email

# Run ETL pipeline
log "Executing ETL pipeline..."

cd "$PROJECT_ROOT"

if "$PYTHON_BIN" "$ETL_SCRIPT"; then
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
