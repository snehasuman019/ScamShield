# ScamShield — AI-Powered Fake Job & Internship Detection System

<p align="center">
  <strong>Detect Fake Job Postings. Protect Job Seekers.</strong>
</p>

<p align="center">
  <a href="https://scamshield-xhxdsyjureq8wpv8zsycfp.streamlit.app/">Live Demo (Streamlit Cloud)</a>
  |
  <a href="https://github.com/snehasuman019/ScamShield">GitHub Repository</a>
</p>

---

## About the Project

ScamShield is an AI-powered web application that identifies potentially fraudulent job and internship postings using machine learning and natural language processing.

The application combines:
- Machine learning classification (Logistic Regression and Random Forest)
- NLP-based text analysis using TF-IDF
- An interactive Streamlit interface
- MongoDB Atlas integration for database operations
- AWS EC2 cloud deployment managed by Terraform and Ansible
- Nagios Core for infrastructure and application monitoring

---

## Live Deployments

| Deployment | URL |
|---|---|
| Streamlit Community Cloud | https://scamshield-xhxdsyjureq8wpv8zsycfp.streamlit.app/ |
| AWS EC2 (ap-south-1) | http://65.0.20.154:8501 |

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.12 |
| Frontend | Streamlit |
| Machine Learning | Scikit-learn (Logistic Regression, Random Forest) |
| NLP | TF-IDF vectorization |
| Data Processing | Pandas, NumPy |
| Database | MongoDB Atlas |
| Database Driver | PyMongo |
| Model Serialization | Joblib |
| Visualization | Plotly |
| Cloud Infrastructure | AWS EC2 (t3.small, ap-south-1) |
| IaC | Terraform |
| Configuration Management | Ansible |
| Monitoring | Nagios Core |
| Version Control | Git, GitHub |

---

## Project Structure

```
ScamShield/
|
|-- analytics/          # Plotly chart generation
|-- data/               # Dataset and download script
|-- database/           # MongoDB connection, CRUD, indexes, validation
|-- infra/
|   |-- ansible/        # Ansible automation
|   |   |-- ansible.cfg
|   |   |-- inventory.ini
|   |   |-- playbook.yml
|   |   |-- group_vars/all.yml
|   |   |-- templates/scamshield.service.j2
|   |-- nagios/         # Nagios host and service definitions
|   |   |-- scamshield.cfg
|   |   |-- nrpe_local.cfg
|   |-- terraform/      # AWS infrastructure as code
|       |-- main.tf
|       |-- .terraform.lock.hcl
|-- ml/                 # ML model training, prediction, preprocessing
|-- models/             # Trained model artifacts (.joblib) and metrics
|-- pages/              # Streamlit multi-page UI
|-- tests/              # Pytest unit tests
|-- app.py              # Main Streamlit application
|-- config.py           # Configuration and environment management
|-- requirements.txt    # Python dependencies
|-- requirements-lock.txt
|-- import_data.py      # Bulk MongoDB import script
|-- ui_utils.py         # Shared UI utilities
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/snehasuman019/ScamShield.git
cd ScamShield
```

### 2. Create virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your MongoDB Atlas URI:

```
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster-url>/?retryWrites=true&w=majority
DB_NAME=scamshield_db
APP_ENV=development
APP_DEBUG=True
```

Never commit your `.env` file.

### 4. Run the application

```bash
streamlit run app.py
```

---

## AWS EC2 Deployment

The application is deployed on an AWS EC2 t3.small instance in `ap-south-1` (Mumbai).

### Architecture

```
Developer Machine (Windows)
        |
        |-- Terraform --> AWS Security Group (SG rules)
        |-- Ansible   --> EC2 Ubuntu Server (configuration + deployment)
        |
        v
  AWS EC2 i-083bfb01091e8ebd5
  Ubuntu 24.04 LTS, t3.small
  65.0.20.154 (ap-south-1b)
        |
        |-- scamshield.service (systemd) --> Streamlit on :8501
        |-- Nagios Core (Apache on :80) --> Monitors host + services
        |
        v
  MongoDB Atlas (cloud.mongodb.net)
  scamshield_db
```

### Connecting to EC2

SSH is provided through EC2 Instance Connect:

```powershell
# From Windows PowerShell — push temporary key (valid 60 seconds)
aws ec2-instance-connect send-ssh-public-key `
  --instance-id i-083bfb01091e8ebd5 `
  --instance-os-user ubuntu `
  --ssh-public-key file://"$env:USERPROFILE\.ssh\scamshield-key-2.pub" `
  --availability-zone ap-south-1b `
  --region ap-south-1 `
  --profile terraform-user

# Then connect
ssh -i "$env:USERPROFILE\.ssh\scamshield-key-2.pem" ubuntu@65.0.20.154
```

---

## Terraform

Terraform manages AWS infrastructure: EC2 instance and security group.

**Directory:** `infra/terraform/`  
**Provider:** AWS (ap-south-1)  
**Profile:** terraform-user

### Commands (run from `infra/terraform/`)

```bash
terraform fmt           # Format configuration
terraform validate      # Validate configuration
terraform plan          # Preview changes (safe, no changes applied)
terraform apply         # Apply changes
```

> **Important:** If your public IP changes, update all three `cidr_blocks` in `main.tf` before running `terraform apply`.

### Resources managed

| Resource | ID | Description |
|---|---|---|
| aws_instance.scamshield | i-083bfb01091e8ebd5 | EC2 t3.small Ubuntu server |
| aws_security_group.scamshield_sg | sg-0acf1c92314b1da78 | Inbound rules for ports 22, 80, 8501 |

---

## Ansible

Ansible automates server configuration and application deployment from your control node (WSL Ubuntu on Windows).

**Directory:** `infra/ansible/`

### Prerequisites (WSL Ubuntu)

```bash
sudo apt update && sudo apt install -y ansible
cp /mnt/c/Users/Sneha/.ssh/scamshield-key-2.pem ~/.ssh/
chmod 600 ~/.ssh/scamshield-key-2.pem
```

### Commands (run from WSL Ubuntu inside `infra/ansible/`)

```bash
# Test connectivity
ansible all -m ping

# Syntax check
ansible-playbook --syntax-check playbook.yml

# Deploy / configure server
ansible-playbook playbook.yml

# Dry run (no changes applied)
ansible-playbook playbook.yml --check
```

### What the playbook does

1. Updates apt packages
2. Installs required system packages
3. Clones or pulls the ScamShield repository
4. Creates Python virtual environment
5. Installs Python dependencies
6. Downloads dataset if not present
7. Deploys `scamshield.service` systemd unit from template
8. Enables and starts the service
9. Verifies the app is responding on port 8501

The playbook is **idempotent** — running it multiple times will not break anything.

> **Security:** The `.env` file with MongoDB credentials is already on EC2. The playbook checks for its presence but never overwrites or reads it.

---

## Nagios Core Monitoring

Nagios Core is installed on the EC2 server and monitors infrastructure and application health.

**Web interface:** http://65.0.20.154/nagios  
**Login:** `nagiosadmin` / (password set during installation)

### Services monitored

| Service | Check | Warning | Critical |
|---|---|---|---|
| Host availability | check-host-alive | — | No response |
| SSH service | check_ssh on port 22 | — | Port closed |
| CPU load | check_load (via NRPE) | > 1.5 | > 3.0 |
| Disk usage | check_disk on / | < 20% free | < 10% free |
| Memory usage | check_mem (via NRPE) | > 80% | > 90% |
| ScamShield app | check_tcp on port 8501 | — | Port closed |

### Monitoring demonstration

```bash
# 1. All services should show OK in Nagios web UI
# 2. Stop ScamShield to trigger CRITICAL alert
sudo systemctl stop scamshield

# 3. Wait 1-2 check intervals, Nagios shows CRITICAL for ScamShield App
# 4. Restart to recover
sudo systemctl start scamshield

# 5. Nagios returns to OK after next check
```

---

## Running Tests

```bash
pytest -v
```

25 unit tests cover configuration, database operations, ML predictions, and preprocessing.

---

## Scope

**DevOps technologies implemented:**
- Terraform: AWS infrastructure provisioning
- Ansible: Server configuration and application deployment automation
- Nagios Core: Infrastructure and application monitoring

**Explicitly excluded:**
- Puppet: Intentionally not implemented for this project
- Docker, Kubernetes, Jenkins: Not required
