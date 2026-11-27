
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region  = "ap-south-1"
  profile = "terraform-user"
}

# Latest Ubuntu 24.04 LTS AMI from Canonical
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Security group for ScamShield server
resource "aws_security_group" "scamshield_sg" {
  name        = "scamshield-security-group"
  description = "Security group for ScamShield"

  # SSH access â€” restricted to administrator IP only
  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["49.156.99.26/32"]
  }

  # Streamlit application â€” restricted to administrator IP only
  ingress {
    description = "Streamlit application"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["49.156.99.26/32"]
  }

  # Nagios web interface â€” restricted to administrator IP only
  ingress {
    description = "Nagios web interface"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["49.156.99.26/32"]
  }

  # Allow all outbound traffic (required for apt, pip, MongoDB Atlas, git)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ScamShield-SG"
  }
}

# EC2 instance â€” ScamShield application server
resource "aws_instance" "scamshield" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.small"

  # key_name must match the key used at instance launch (scamshield-key).
  # SSH access is provided via EC2 Instance Connect using scamshield-key-2.pem.
  # Do NOT change this value â€” it would force instance replacement.
  key_name = "scamshield-key"

  vpc_security_group_ids = [aws_security_group.scamshield_sg.id]

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  tags = {
    Name        = "ScamShield-Server"
    Environment = "Development"
    Project     = "ScamShield"
  }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.scamshield.id
}

output "instance_public_ip" {
  description = "Public IP address of the ScamShield server"
  value       = aws_instance.scamshield.public_ip
}

output "instance_public_dns" {
  description = "Public DNS of the ScamShield server"
  value       = aws_instance.scamshield.public_dns
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.scamshield_sg.id
}
