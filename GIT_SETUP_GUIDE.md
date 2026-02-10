# Git Repository Setup Guide

## ✅ Local Repository Created

Your project is now under version control! Here's what was done:

- ✓ Git repository initialized
- ✓ `.gitignore` configured (excludes logs, data files, .env)
- ✓ 21 files staged and committed
- ✓ **3,882 lines** of code, documentation, and configuration
- ✓ Commit hash: `0d223aa`

---

## 🚀 Push to Remote Repository

### Option 1: GitHub

1. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Repository name: `fibre-data-etl-pipeline`
   - Don't initialize with README (we already have one)
   - Click "Create repository"

2. **Push your code:**
   ```bash
   cd /home/habib/fibre_data_project/projet-fibre-forecast
   
   # Add GitHub as remote
   git remote add origin https://github.com/YOUR_USERNAME/fibre-data-etl-pipeline.git
   
   # Push to GitHub
   git push -u origin main
   ```

3. **Using SSH (recommended):**
   ```bash
   # If you have SSH keys set up
   git remote add origin git@github.com:YOUR_USERNAME/fibre-data-etl-pipeline.git
   git push -u origin main
   ```

### Option 2: GitLab

1. **Create a new project on GitLab:**
   - Go to https://gitlab.com/projects/new
   - Project name: `fibre-data-etl-pipeline`
   - Visibility: Private (recommended)
   - Don't initialize with README
   - Click "Create project"

2. **Push your code:**
   ```bash
   cd /home/habib/fibre_data_project/projet-fibre-forecast
   
   # Add GitLab as remote
   git remote add origin https://gitlab.com/YOUR_USERNAME/fibre-data-etl-pipeline.git
   
   # Push to GitLab
   git push -u origin main
   ```

### Option 3: Bitbucket

1. **Create a new repository on Bitbucket:**
   - Go to https://bitbucket.org/repo/create
   - Repository name: `fibre-data-etl-pipeline`
   - Access level: Private
   - Click "Create repository"

2. **Push your code:**
   ```bash
   cd /home/habib/fibre_data_project/projet-fibre-forecast
   
   # Add Bitbucket as remote
   git remote add origin https://YOUR_USERNAME@bitbucket.org/YOUR_USERNAME/fibre-data-etl-pipeline.git
   
   # Push to Bitbucket
   git push -u origin main
   ```

### Option 4: Self-Hosted Git Server

```bash
cd /home/habib/fibre_data_project/projet-fibre-forecast

# Add your server as remote
git remote add origin git@your-server.com:path/to/repo.git

# Push to server
git push -u origin main
```

---

## 🔐 Authentication

### HTTPS (Username/Password or Token)

When pushing via HTTPS, you'll be prompted for credentials:
- **Username:** Your Git service username
- **Password:** 
  - GitHub: Use a Personal Access Token (not your password)
  - GitLab: Use Personal Access Token or password
  - Bitbucket: Use App Password

**Create tokens:**
- **GitHub:** Settings → Developer settings → Personal access tokens → Tokens (classic)
- **GitLab:** Preferences → Access Tokens
- **Bitbucket:** Personal settings → App passwords

### SSH (Recommended)

More secure and no password prompts:

1. **Generate SSH key (if you don't have one):**
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

2. **Add public key to your Git service:**
   ```bash
   # Copy your public key
   cat ~/.ssh/id_ed25519.pub
   ```
   
   Then add it to:
   - **GitHub:** Settings → SSH and GPG keys
   - **GitLab:** Preferences → SSH Keys
   - **Bitbucket:** Personal settings → SSH keys

3. **Test connection:**
   ```bash
   ssh -T git@github.com
   # or
   ssh -T git@gitlab.com
   ```

---

## 📊 Repository Status

```bash
# Check current status
git status

# View commit history
git log --oneline

# View remote repositories
git remote -v
```

---

## 🔄 Daily Workflow

After making changes:

```bash
# Check what changed
git status

# Stage changes
git add .

# Or stage specific files
git add src/etl/config.py

# Commit with message
git commit -m "Description of changes"

# Push to remote
git push
```

---

## 📝 Useful Git Commands

```bash
# View changes before staging
git diff

# View staged changes
git diff --staged

# Undo changes in working directory
git checkout -- filename.py

# Unstage files
git reset HEAD filename.py

# View commit history with details
git log --graph --oneline --all

# Create a new branch
git checkout -b feature/new-feature

# Switch branches
git checkout main

# Show remote URLs
git remote -v

# Change remote URL
git remote set-url origin new-url
```

---

## 🏷️ Tagging Releases

```bash
# Create a tag for version 1.0
git tag -a v1.0.0 -m "Initial release: Production-ready ETL Pipeline"

# Push tags to remote
git push --tags

# List all tags
git tag -l
```

---

## 📦 What's Tracked in the Repository

**Included (tracked by Git):**
- ✓ All Python source code (`src/etl/*.py`)
- ✓ Database schema (`docker/init-scripts/*.sql`)
- ✓ Docker configuration (`docker-compose.yml`)
- ✓ Documentation (`.md` files)
- ✓ Configuration templates (`.env.example`)
- ✓ Automation scripts (`Makefile`, `watch_etl.sh`)
- ✓ Directory structure (`.gitkeep` files)

**Excluded (in .gitignore):**
- ✗ Environment variables (`.env`)
- ✗ Data files (`data/*/*.csv`)
- ✗ Log files (`logs/*.log`)
- ✗ Python cache (`__pycache__/`, `*.pyc`)
- ✗ Virtual environments (`venv/`, `env/`)
- ✗ IDE settings (`.vscode/`, `.idea/`)

---

## 🌿 Branching Strategy (Recommended)

```bash
# Main branch (production-ready)
main

# Development branch
git checkout -b develop

# Feature branches
git checkout -b feature/add-forecasting
git checkout -b feature/dashboard-integration

# Bug fixes
git checkout -b bugfix/fix-date-parsing

# Merge back to main
git checkout main
git merge feature/add-forecasting
git push
```

---

## 🚨 Common Issues

### Issue: Remote already exists
```bash
# Remove existing remote
git remote remove origin

# Add new remote
git remote add origin YOUR_NEW_URL
```

### Issue: Authentication failed
```bash
# Use SSH instead of HTTPS
git remote set-url origin git@github.com:USERNAME/REPO.git

# Or update Git credentials
git config --global credential.helper store
```

### Issue: Large files error
```bash
# Files over 100MB need Git LFS
git lfs install
git lfs track "*.csv"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

---

## 📤 Quick Push Commands

**After creating remote repository:**

```bash
# GitHub
git remote add origin https://github.com/YOUR_USERNAME/fibre-data-etl-pipeline.git
git push -u origin main

# GitLab
git remote add origin https://gitlab.com/YOUR_USERNAME/fibre-data-etl-pipeline.git
git push -u origin main
```

**Subsequent pushes:**
```bash
git push
```

---

## ✅ Next Steps

1. **Create remote repository** on GitHub/GitLab/Bitbucket
2. **Add remote origin** using commands above
3. **Push your code** with `git push -u origin main`
4. **Add collaborators** (optional)
5. **Set up CI/CD** (optional, for automated testing)

---

## 🔗 Current Repository Info

- **Branch:** `main`
- **Last Commit:** `0d223aa`
- **Files Tracked:** 21
- **Total Lines:** 3,882
- **Last Commit Message:** "Initial commit: Complete ETL Pipeline for Fibre Data"

---

## 📞 Support

For Git help:
```bash
git help
git help <command>  # e.g., git help push
```

Online resources:
- GitHub Docs: https://docs.github.com
- GitLab Docs: https://docs.gitlab.com
- Git Documentation: https://git-scm.com/doc

---

**Your code is ready to push! Choose a Git hosting service and follow the steps above.** 🚀
