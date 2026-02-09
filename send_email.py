#!/usr/bin/env python3
"""
Simple email notification script using SMTP
"""
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os

def send_email(to_email, subject, body, from_email=None):
    """
    Send email using SMTP
    Supports:
    - Gmail with app password
    - Local SMTP relay
    - External SMTP server
    """
    
    if from_email is None:
        from_email = os.getenv('FROM_EMAIL', 'ETL-notify@local')
    
    # Try local SMTP first (no auth)
    try:
        with smtplib.SMTP('localhost', 25, timeout=5) as server:
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            server.send_message(msg)
            print(f"Email sent via local SMTP to {to_email}")
            return True
    except Exception as e:
        pass  # Try next method
    
    # Try Gmail SMTP (requires app password)
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    
    if smtp_server and smtp_user and smtp_password:
        try:
            msg = MIMEMultipart()
            msg['From'] = from_email if from_email != 'ETL-notify@local' else smtp_user
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
                
            print(f"Email sent via {smtp_server} to {to_email}")
            return True
        except Exception as e:
            print(f"SMTP error: {e}", file=sys.stderr)
            return False
    
    print("No SMTP configuration available. Email not sent.", file=sys.stderr)
    print("To enable email, either:", file=sys.stderr)
    print("  1. Install local mail server: sudo apt-get install postfix mailutils", file=sys.stderr)
    print("  2. Configure SMTP env vars: SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD", file=sys.stderr)
    return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: send_email.py <to_email> <subject> [body]")
        sys.exit(1)
    
    to = sys.argv[1]
    subject = sys.argv[2]
    body = sys.argv[3] if len(sys.argv) > 3 else ""
    
    success = send_email(to, subject, body)
    sys.exit(0 if success else 1)
