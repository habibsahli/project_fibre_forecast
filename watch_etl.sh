#!/bin/bash
# Watch landing/ for new CSV files and auto-run ETL

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ETL_SCRIPT="$PROJECT_ROOT/src/etl/etl_main.py"
DOCKER_COMPOSE="$PROJECT_ROOT/docker/docker-compose.yml"
LANDING_DIR="$PROJECT_ROOT/data/landing"
LOCK_FILE="/tmp/etl_watch.lock"
INOTIFY_BIN="${INOTIFY_BIN:-inotifywait}"
DEBOUNCE_SECONDS="${DEBOUNCE_SECONDS:-10}"
NOTIFY_EMAIL="${NOTIFY_EMAIL:-}"
FROM_EMAIL="${FROM_EMAIL:-ETL-notify@local}"
SUBJECT_PREFIX="${SUBJECT_PREFIX:-ETL Launch}"
HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

require_inotify() {
    if ! command -v "$INOTIFY_BIN" >/dev/null 2>&1; then
        log "ERROR: inotifywait not found. Install inotify-tools."
        log "  Debian/Ubuntu: sudo apt-get install inotify-tools"
        log "  RHEL/CentOS:   sudo yum install inotify-tools"
        exit 1
    fi
}

send_launch_email() {
    if [ -z "$NOTIFY_EMAIL" ]; then
        return 0
    fi

    subject="$SUBJECT_PREFIX: $HOSTNAME_SHORT"
    body="ETL pipeline launch detected on $HOSTNAME_SHORT at $(date '+%Y-%m-%d %H:%M:%S')."

    # Try Python email script first
    if [ -f "$PROJECT_ROOT/send_email.py" ] && [ -n "$PYTHON_BIN" ]; then
        if "$PYTHON_BIN" "$PROJECT_ROOT/send_email.py" "$NOTIFY_EMAIL" "$subject" "$body" 2>&1; then
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

ensure_docker() {
    if ! docker ps >/dev/null 2>&1; then
        log "ERROR: Docker is not running"
        return 1
    fi

    if ! docker-compose -f "$DOCKER_COMPOSE" ps postgres | grep -q "Up"; then
        log "PostgreSQL container not running. Starting..."
        docker-compose -f "$DOCKER_COMPOSE" up -d
        sleep 10
    fi
}

run_etl() {
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        log "ETL already running. Skipping this event."
        return 0
    fi

    send_launch_email
    log "Executing ETL pipeline..."

    if ! ensure_docker; then
        log "ERROR: Docker not ready. ETL not started."
        return 1
    fi

    if (cd "$PROJECT_ROOT" && "$PYTHON_BIN" "$ETL_SCRIPT"); then
        log "SUCCESS: ETL pipeline completed"
    else
        log "ERROR: ETL pipeline failed"
        return 1
    fi
}

require_inotify

log "Watching for new CSV files in: $LANDING_DIR"
log "Debounce window: ${DEBOUNCE_SECONDS}s"

while true; do
    first_path="$($INOTIFY_BIN -e close_write -e moved_to --format '%w%f' "$LANDING_DIR")"
    case "$first_path" in
        *.csv|*.CSV)
            log "Detected new CSV: $first_path"
            ;;
        *)
            log "Ignored non-CSV: $first_path"
            continue
            ;;
    esac

    while true; do
        if next_path="$($INOTIFY_BIN -e close_write -e moved_to -t "$DEBOUNCE_SECONDS" --format '%w%f' "$LANDING_DIR")"; then
            case "$next_path" in
                *.csv|*.CSV)
                    log "Detected additional CSV: $next_path"
                    ;;
                *)
                    log "Ignored non-CSV: $next_path"
                    ;;
            esac
        else
            break
        fi
    done

    run_etl
done
