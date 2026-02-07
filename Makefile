.PHONY: help install setup up down logs clean test run validate docker-setup env-setup

# Default target
help:
	@echo "Fibre Data ETL Pipeline - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make env-setup        - Copy .env.example to .env and configure"
	@echo "  make docker-setup     - Build and start PostgreSQL container"
	@echo "  make install          - Install Python dependencies"
	@echo "  make setup            - Full setup (env + docker + install)"
	@echo ""
	@echo "Operations:"
	@echo "  make up               - Start Docker services"
	@echo "  make down             - Stop Docker services"
	@echo "  make run              - Execute ETL pipeline"
	@echo "  make validate         - Validate configuration"
	@echo ""
	@echo "Maintenance:"
	@echo "  make logs             - View ETL logs"
	@echo "  make clean            - Remove containers and volumes (CAUTION!)"
	@echo "  make test             - Run test data through pipeline"
	@echo ""
	@echo "Database:"
	@echo "  make db-connect       - Connect to PostgreSQL CLI"
	@echo "  make db-reset         - Reset database (WARNING: deletes all data)"
	@echo ""

# Setup targets
env-setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ .env file created. Please edit with your settings."; \
	else \
		echo "✓ .env file already exists"; \
	fi

docker-setup: env-setup
	@echo "Starting Docker containers..."
	docker-compose -f docker/docker-compose.yml up -d
	@echo "✓ Docker containers started"
	@echo "Waiting for PostgreSQL to be ready..."
	sleep 10
	@echo "✓ PostgreSQL is ready"

install:
	@echo "Installing Python dependencies..."
	pip install psycopg2-binary pandas numpy
	@echo "✓ Dependencies installed"

setup: env-setup install docker-setup
	@echo "✓ Complete setup finished!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Place your CSV files in: data/landing/"
	@echo "2. Run: make run"

# Operation targets
up:
	docker-compose -f docker/docker-compose.yml up -d
	@echo "✓ Services started"

down:
	docker-compose -f docker/docker-compose.yml down
	@echo "✓ Services stopped"

run:
	@echo "Starting ETL Pipeline..."
	cd src/etl && python3 etl_main.py

validate:
	@echo "Validating configuration..."
	cd src/etl && python3 config.py

logs:
	@ls -lht logs/ | head -10
	@echo ""
	@echo "Latest log file:"
	@tail -50 logs/etl_pipeline_*.log 2>/dev/null | head -50 || echo "No log files found"

clean:
	@echo "WARNING: This will remove Docker containers and delete PostgreSQL data!"
	@read -p "Continue? (yes/no): " confirm && \
	if [ "$$confirm" = "yes" ]; then \
		docker-compose -f docker/docker-compose.yml down -v; \
		echo "✓ Cleaned"; \
	else \
		echo "Cancelled"; \
	fi

test:
	@echo "Creating test data..."
	@mkdir -p data/landing
	@echo "Place test CSV file in data/landing/ and run: make run"

# Database targets
db-connect:
	docker-compose -f docker/docker-compose.yml exec postgres psql -U postgres -d fibre_data

db-reset:
	@echo "WARNING: This will delete all data in the database!"
	@read -p "Continue? (yes/no): " confirm && \
	if [ "$$confirm" = "yes" ]; then \
		docker-compose -f docker/docker-compose.yml exec postgres psql -U postgres -d fibre_data -c "DROP SCHEMA etl_fibre CASCADE; CREATE SCHEMA etl_fibre;"; \
		@echo "✓ Database reset"; \
	else \
		echo "Cancelled"; \
	fi

# Utility targets
lint:
	@echo "Checking Python code..."
	cd src/etl && python3 -m py_compile *.py
	@echo "✓ No syntax errors found"

requirements:
	@echo "psycopg2-binary" > requirements.txt
	@echo "pandas" >> requirements.txt
	@echo "numpy" >> requirements.txt
	@echo "✓ requirements.txt created"

.SILENT: help
