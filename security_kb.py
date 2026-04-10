"""
SecurityKnowledgeBase — Pre-indexed IaC security rules for RAG augmentation.

Bundles a curated set of OWASP, CIS Benchmark, and cloud-provider best-practice
rules as text documents.  These are embedded and stored in a *separate* FAISS
index alongside the code index, so both authoritative security guidance and
code chunks are retrieved for every analysis query.

Built-in rule categories
------------------------
• OWASP Top-10 for Cloud / IaC
• CIS AWS Foundations Benchmark v1.5
• CIS Kubernetes Benchmark v1.8
• Azure Security Benchmark
• CIS Azure IAM / Managed Identity rules
• GCP Security Best Practices
• General IaC / secrets patterns

Persistence
-----------
Rules are embedded once and saved to ``<index_path>.faiss`` +
``<index_path>.db``.  Subsequent runs load the saved index directly,
avoiding re-embedding.

Usage
-----
    kb = SecurityKnowledgeBase(embedder=my_embedder)
    kb.load_or_build()                          # idempotent
    results = kb.search(query_vec, top_k=3)     # → List[SecurityRuleResult]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in security rule corpus
# ---------------------------------------------------------------------------

_BUILT_IN_RULES: List[Dict[str, str]] = [
    # ── OWASP ──────────────────────────────────────────────────────────────
    {
        "id":          "owasp-iac-01",
        "category":    "owasp",
        "severity":    "CRITICAL",
        "title":       "Hardcoded Credentials and Secrets",
        "description": (
            "Hardcoded passwords, API keys, access tokens, secret keys, or other "
            "sensitive values embedded directly in IaC code expose credentials to "
            "anyone with repository access. Maps to OWASP A02:2021 – Cryptographic "
            "Failures and CWE-798 (Use of Hard-coded Credentials)."
        ),
        "indicators":  (
            "password = , secret = , api_key = , access_key = , private_key = , "
            "token = , credential, hardcoded, plaintext, base64 encoded secret, "
            "db_password, database_password, master_password"
        ),
        "remediation": (
            "Use a secrets management solution: AWS Secrets Manager, HashiCorp Vault, "
            "Azure Key Vault, or GCP Secret Manager. Reference secrets via "
            "environment variables or secret store data sources. Never commit "
            "credential values to source control."
        ),
        "references":  "CWE-798, OWASP A02:2021, CIS AWS 1.12",
    },
    {
        "id":          "owasp-iac-02",
        "category":    "owasp",
        "severity":    "HIGH",
        "title":       "Missing Encryption at Rest",
        "description": (
            "Storage resources (S3 buckets, EBS volumes, RDS instances, Azure Storage "
            "Accounts, GCP disks) without server-side encryption store data in "
            "plaintext. If accessed by an adversary, data is immediately readable. "
            "Maps to OWASP A02:2021 – Cryptographic Failures, CWE-311."
        ),
        "indicators":  (
            "server_side_encryption = false, encrypt = false, "
            "storage_encrypted = false, encrypted = false, "
            "kms_key_id missing, encryption disabled"
        ),
        "remediation": (
            "Enable server-side encryption using managed keys (SSE-S3, SSE-KMS, "
            "AES-256). Prefer customer-managed KMS keys (CMK) for sensitive data. "
            "Set `server_side_encryption_configuration` in S3, `encrypted = true` "
            "in RDS/EBS, or `enable_disk_encryption = true` in Azure/GCP."
        ),
        "references":  "CWE-311, OWASP A02:2021, CIS AWS 2.3, CIS AWS 2.7",
    },
    {
        "id":          "owasp-iac-03",
        "category":    "owasp",
        "severity":    "HIGH",
        "title":       "Missing Encryption in Transit (TLS/SSL Disabled)",
        "description": (
            "Disabling TLS/SSL or allowing insecure protocols (HTTP, FTP, Telnet) "
            "exposes data to interception. This applies to load balancers, databases, "
            "messaging queues, and APIs. Maps to OWASP A02:2021, CWE-319."
        ),
        "indicators":  (
            "ssl_enabled = false, tls = false, insecure = true, "
            "http only, protocol = HTTP, min_tls_version = TLS1, "
            "enforce_https = false, require_secure_transport = OFF"
        ),
        "remediation": (
            "Enable TLS 1.2 or higher on all endpoints. Set minimum TLS version "
            "to TLSv1.2. Configure listeners to redirect HTTP → HTTPS. "
            "Set `require_secure_transport = ON` for RDS. Use "
            "`enforce_https = true` for Azure Storage."
        ),
        "references":  "CWE-319, OWASP A02:2021, CIS AWS 4.3",
    },
    {
        "id":          "owasp-iac-04",
        "category":    "owasp",
        "severity":    "CRITICAL",
        "title":       "Overly Permissive IAM Policies (Wildcard Actions)",
        "description": (
            "IAM policies granting Action: \"*\" or Resource: \"*\" provide "
            "unrestricted access and violate the principle of least privilege. "
            "A compromised principal with wildcard permissions can read, modify, "
            "or delete any resource. Maps to OWASP A01:2021 – Broken Access Control, "
            "CWE-732."
        ),
        "indicators":  (
            'Action": "*", Action": "iam:*", Resource": "*", '
            "NotAction, AdministratorAccess, FullAccess, PowerUserAccess, "
            "wildcard permission, allow all"
        ),
        "remediation": (
            "Apply least-privilege: enumerate only the specific actions required "
            "(e.g., s3:GetObject, s3:PutObject). Scope resources to specific ARNs. "
            "Use IAM Access Analyzer to detect and remediate over-permissive policies. "
            "Avoid wildcards in production. Use SCPs to enforce org-level boundaries."
        ),
        "references":  "CWE-732, OWASP A01:2021, CIS AWS 1.16, CIS AWS 1.22",
    },
    {
        "id":          "owasp-iac-05",
        "category":    "owasp",
        "severity":    "CRITICAL",
        "title":       "Publicly Accessible Cloud Storage",
        "description": (
            "S3 buckets, Azure Blob containers, or GCP Storage buckets configured "
            "with public ACLs (\"public-read\", \"public-read-write\") or disabled "
            "Block Public Access settings expose data to the internet. "
            "Maps to OWASP A01:2021 – Broken Access Control, CWE-284."
        ),
        "indicators":  (
            "acl = public-read, acl = public-read-write, "
            "block_public_acls = false, block_public_policy = false, "
            "ignore_public_acls = false, restrict_public_buckets = false, "
            "AllowPublic, public-access"
        ),
        "remediation": (
            "Set `block_public_acls = true`, `block_public_policy = true`, "
            "`ignore_public_acls = true`, `restrict_public_buckets = true` on all S3 "
            "buckets. Use private ACL. Enable S3 Block Public Access at the account "
            "level. For Azure, set `allow_blob_public_access = false`."
        ),
        "references":  "CWE-284, OWASP A01:2021, CIS AWS 2.1, CIS AWS 2.2",
    },
    {
        "id":          "owasp-iac-06",
        "category":    "owasp",
        "severity":    "HIGH",
        "title":       "Missing Audit Logging and Monitoring",
        "description": (
            "Disabling CloudTrail, VPC Flow Logs, Azure Monitor, or GCP Cloud Audit "
            "Logs removes the ability to detect and investigate security incidents. "
            "Maps to OWASP A09:2021 – Security Logging and Monitoring Failures, "
            "CWE-778."
        ),
        "indicators":  (
            "enable_log_file_validation = false, is_logging = false, "
            "flow_logs_enabled = false, audit_log = disabled, "
            "cloudtrail disabled, logging = false"
        ),
        "remediation": (
            "Enable AWS CloudTrail in all regions with log file validation. "
            "Enable VPC Flow Logs. Enable AWS Config. "
            "Set `enable_log_file_validation = true` in CloudTrail resources. "
            "Ship logs to a centralised, immutable S3 bucket with MFA delete enabled."
        ),
        "references":  "CWE-778, OWASP A09:2021, CIS AWS 3.1–3.14",
    },
    # ── CIS AWS ────────────────────────────────────────────────────────────
    {
        "id":          "cis-aws-01",
        "category":    "cis_aws",
        "severity":    "HIGH",
        "title":       "Open Security Group — All Traffic Allowed (0.0.0.0/0)",
        "description": (
            "Security group ingress rules permitting all traffic from 0.0.0.0/0 "
            "(::/0 for IPv6) expose compute resources to the entire internet. "
            "Common misconfigurations include open SSH (port 22), RDP (port 3389), "
            "and wildcard port ranges (-1 to -1). CIS AWS 4.1, 4.2."
        ),
        "indicators":  (
            "cidr_blocks = 0.0.0.0/0, ipv6_cidr_blocks = ::/0, "
            "from_port = 0, to_port = 0, from_port = 22, to_port = 22, "
            "from_port = 3389, protocol = -1, all traffic ingress"
        ),
        "remediation": (
            "Restrict ingress to known IP ranges. Remove 0.0.0.0/0 from security "
            "group ingress rules. Use bastion hosts or AWS Systems Manager Session "
            "Manager instead of exposing SSH/RDP. Apply Security Group rule auditing "
            "via AWS Config rule `restricted-ssh`."
        ),
        "references":  "CIS AWS 4.1, CIS AWS 4.2, CWE-284",
    },
    {
        "id":          "cis-aws-02",
        "category":    "cis_aws",
        "severity":    "HIGH",
        "title":       "Root Account Access Keys Present",
        "description": (
            "Creating or using access keys for the AWS root account is highly "
            "dangerous. Root credentials have unrestricted access. If compromised, "
            "the entire account is at risk. CIS AWS 1.4."
        ),
        "indicators":  (
            "root account access key, aws_access_key_id for root, "
            "create_access_key for root user, root credentials"
        ),
        "remediation": (
            "Delete all root account access keys. Enable MFA on the root account. "
            "Use IAM users and roles with least-privilege for all operations. "
            "Monitor root account usage via CloudWatch alarms."
        ),
        "references":  "CIS AWS 1.4, CIS AWS 1.5",
    },
    {
        "id":          "cis-aws-03",
        "category":    "cis_aws",
        "severity":    "MEDIUM",
        "title":       "S3 Bucket Versioning Disabled",
        "description": (
            "Without versioning, deleted or overwritten objects cannot be recovered. "
            "This facilitates data destruction in ransomware or accidental deletion "
            "scenarios. CIS AWS 2.9."
        ),
        "indicators":  (
            "versioning { enabled = false }, versioning disabled, "
            "versioning_configuration status = Suspended"
        ),
        "remediation": (
            "Enable S3 versioning on all buckets containing sensitive or critical "
            "data. Configure lifecycle rules to expire old versions. Enable MFA "
            "delete for buckets storing audit logs."
        ),
        "references":  "CIS AWS 2.9, AWS Best Practices",
    },
    {
        "id":          "cis-aws-04",
        "category":    "cis_aws",
        "severity":    "HIGH",
        "title":       "CloudTrail Not Enabled in All Regions",
        "description": (
            "CloudTrail must be enabled in all regions to capture API calls globally. "
            "Single-region trails miss activity in other regions. CIS AWS 3.1."
        ),
        "indicators":  (
            "is_multi_region_trail = false, include_global_service_events = false, "
            "cloudtrail not all regions, trail disabled"
        ),
        "remediation": (
            "Set `is_multi_region_trail = true` and `include_global_service_events = true`. "
            "Enable CloudTrail in every region. Use AWS Organizations CloudTrail for "
            "centralized multi-account logging."
        ),
        "references":  "CIS AWS 3.1, CIS AWS 3.2",
    },
    {
        "id":          "cis-aws-05",
        "category":    "cis_aws",
        "severity":    "HIGH",
        "title":       "RDS Instance Publicly Accessible",
        "description": (
            "Setting `publicly_accessible = true` on an RDS instance exposes the "
            "database endpoint to the internet. Combined with weak credentials or "
            "permissive security groups, this leads to data breach. CIS AWS 2.3."
        ),
        "indicators":  (
            "publicly_accessible = true, rds public, database public endpoint, "
            "multi_az = false, skip_final_snapshot = true"
        ),
        "remediation": (
            "Set `publicly_accessible = false`. Place RDS instances in private subnets. "
            "Use security groups to restrict access to application servers only. "
            "Enable Multi-AZ for production databases."
        ),
        "references":  "CIS AWS 2.3, OWASP A01:2021",
    },
    {
        "id":          "cis-aws-06",
        "category":    "cis_aws",
        "severity":    "MEDIUM",
        "title":       "KMS Key Rotation Disabled",
        "description": (
            "KMS keys without automatic rotation increase the blast radius if a key "
            "is compromised. CIS AWS 3.8 requires annual rotation."
        ),
        "indicators":  (
            "enable_key_rotation = false, key_rotation disabled, "
            "kms rotation not enabled"
        ),
        "remediation": (
            "Set `enable_key_rotation = true` on all KMS keys used for data "
            "encryption. This enables automatic annual rotation."
        ),
        "references":  "CIS AWS 3.8",
    },
    # ── CIS Kubernetes ─────────────────────────────────────────────────────
    {
        "id":          "cis-k8s-01",
        "category":    "cis_k8s",
        "severity":    "CRITICAL",
        "title":       "Kubernetes Container Running as Root (UID 0)",
        "description": (
            "Containers running as root (UID 0) can exploit container escape "
            "vulnerabilities to gain host root access. CIS Kubernetes 5.2.6."
        ),
        "indicators":  (
            "runAsUser: 0, runAsNonRoot: false, allowPrivilegeEscalation: true, "
            "user: root, securityContext missing"
        ),
        "remediation": (
            "Set `runAsNonRoot: true` and `runAsUser: <non-zero>` in the container "
            "securityContext. Use `allowPrivilegeEscalation: false`. "
            "Apply Pod Security Admission (restricted profile) or PSP."
        ),
        "references":  "CIS K8s 5.2.6, CWE-250",
    },
    {
        "id":          "cis-k8s-02",
        "category":    "cis_k8s",
        "severity":    "CRITICAL",
        "title":       "Privileged Kubernetes Container",
        "description": (
            "Setting `privileged: true` gives a container all Linux capabilities "
            "and access to the host device namespace, effectively granting host root. "
            "CIS Kubernetes 5.2.1."
        ),
        "indicators":  (
            "privileged: true, capabilities add: ALL, "
            "allowPrivilegedContainers: true, securityContext privileged"
        ),
        "remediation": (
            "Remove `privileged: true`. Drop all capabilities and add only those "
            "strictly required: `capabilities: drop: [ALL], add: [NET_BIND_SERVICE]`. "
            "Use `seccompProfile: type: RuntimeDefault`."
        ),
        "references":  "CIS K8s 5.2.1, CWE-250",
    },
    {
        "id":          "cis-k8s-03",
        "category":    "cis_k8s",
        "severity":    "HIGH",
        "title":       "Kubernetes Workload Using Host Network or PID Namespace",
        "description": (
            "Setting `hostNetwork: true` or `hostPID: true` breaks pod isolation. "
            "Containers can see host-level network traffic or processes. "
            "CIS Kubernetes 5.2.4."
        ),
        "indicators":  (
            "hostNetwork: true, hostPID: true, hostIPC: true, "
            "shareProcessNamespace: true"
        ),
        "remediation": (
            "Set `hostNetwork: false`, `hostPID: false`, `hostIPC: false`. "
            "Use ClusterIP Services instead of host networking. "
            "Apply the restricted Pod Security Standard."
        ),
        "references":  "CIS K8s 5.2.4, CIS K8s 5.2.3",
    },
    {
        "id":          "cis-k8s-04",
        "category":    "cis_k8s",
        "severity":    "HIGH",
        "title":       "Missing Kubernetes Resource Limits",
        "description": (
            "Containers without CPU and memory limits can consume all node resources "
            "(resource exhaustion / DoS). Requests and limits must both be set. "
            "CIS Kubernetes 5.2.9."
        ),
        "indicators":  (
            "resources: {}, resources limits missing, resources requests missing, "
            "cpu limit not set, memory limit not set"
        ),
        "remediation": (
            "Define both `resources.requests` and `resources.limits` for every "
            "container. Use LimitRange objects to enforce defaults at the namespace "
            "level. Apply ResourceQuota to cap namespace-level resource usage."
        ),
        "references":  "CIS K8s 5.2.9, OWASP A05:2021",
    },
    {
        "id":          "cis-k8s-05",
        "category":    "cis_k8s",
        "severity":    "MEDIUM",
        "title":       "Kubernetes Secrets Exposed as Environment Variables",
        "description": (
            "Storing secrets directly in environment variables (instead of "
            "secretKeyRef) makes them visible in pod specs, logs, and "
            "`kubectl describe pod` output."
        ),
        "indicators":  (
            "env value: hardcoded secret, env password: plaintext, "
            "env API_KEY direct value, secret in env var not secretKeyRef"
        ),
        "remediation": (
            "Use `secretKeyRef` to reference Kubernetes Secrets. Consider an "
            "external secrets operator (ESO, Vault Agent, AWS ASCP). "
            "Never commit Secret manifests with base64 data to source control."
        ),
        "references":  "CWE-312, CIS K8s 5.4.1",
    },
    # ── Azure ──────────────────────────────────────────────────────────────
    {
        "id":          "azure-01",
        "category":    "azure",
        "severity":    "HIGH",
        "title":       "Azure Storage Account Allows Public Blob Access",
        "description": (
            "Setting `allow_blob_public_access = true` on an Azure Storage Account "
            "permits unauthenticated access to blobs in public containers. "
            "Azure Security Benchmark NS-2."
        ),
        "indicators":  (
            "allow_blob_public_access = true, public blob access, "
            "min_tls_version = TLS1_0, enable_https_traffic_only = false"
        ),
        "remediation": (
            "Set `allow_blob_public_access = false`. Set `enable_https_traffic_only "
            "= true`. Set `min_tls_version = TLS1_2`. Enable Azure Defender for "
            "Storage."
        ),
        "references":  "Azure Security Benchmark NS-2, CIS Azure 3.1",
    },
    {
        "id":          "azure-02",
        "category":    "azure",
        "severity":    "HIGH",
        "title":       "Azure Network Security Group Allows Inbound Internet Traffic",
        "description": (
            "NSG rules with `source_address_prefix = '*'` or `'Internet'` and "
            "destination ports 22 (SSH) or 3389 (RDP) expose VMs to brute-force "
            "attacks. CIS Azure 6.1, 6.2."
        ),
        "indicators":  (
            "source_address_prefix = *, source_address_prefix = Internet, "
            "destination_port_range = 22, destination_port_range = 3389, "
            "access = Allow, direction = Inbound"
        ),
        "remediation": (
            "Replace * source with specific trusted IP ranges. Use Azure Bastion "
            "for admin access. Disable public SSH/RDP. Apply Just-in-Time VM access."
        ),
        "references":  "CIS Azure 6.1, CIS Azure 6.2",
    },
    # ── GCP ────────────────────────────────────────────────────────────────
    {
        "id":          "gcp-01",
        "category":    "gcp",
        "severity":    "CRITICAL",
        "title":       "GCP Storage Bucket Publicly Accessible (allUsers / allAuthenticatedUsers)",
        "description": (
            "Granting `allUsers` or `allAuthenticatedUsers` IAM bindings to a GCP "
            "Storage bucket makes it publicly readable or writable. "
            "CIS GCP 5.1."
        ),
        "indicators":  (
            "member = allUsers, member = allAuthenticatedUsers, "
            "public bucket, IAM allUsers binding"
        ),
        "remediation": (
            "Remove `allUsers` and `allAuthenticatedUsers` bindings. Enable "
            "Uniform Bucket-Level Access. Use signed URLs for temporary public access."
        ),
        "references":  "CIS GCP 5.1, CWE-284",
    },
    {
        "id":          "gcp-02",
        "category":    "gcp",
        "severity":    "HIGH",
        "title":       "GCP VM Instance with Public IP and No Firewall Restriction",
        "description": (
            "GCP Compute Engine instances with `access_type = EXTERNAL` or an "
            "ephemeral external IP, combined with permissive firewall rules, are "
            "directly reachable from the internet."
        ),
        "indicators":  (
            "access_config nat_ip, ephemeral external IP, "
            "firewall allow 0.0.0.0/0, google_compute_firewall allow all"
        ),
        "remediation": (
            "Remove external IP addresses. Use Cloud NAT for outbound connectivity. "
            "Use IAP (Identity-Aware Proxy) for admin access. Restrict firewall rules "
            "to specific source ranges."
        ),
        "references":  "CIS GCP 4.6, CWE-284",
    },
    # ── General IaC Patterns ────────────────────────────────────────────────
    {
        "id":          "general-01",
        "category":    "general",
        "severity":    "HIGH",
        "title":       "Default or Weak Admin Password",
        "description": (
            "Using default passwords (\"admin\", \"password\", \"changeme\", "
            "\"12345\") for databases, message brokers, or admin portals is "
            "trivially exploitable. CWE-521."
        ),
        "indicators":  (
            "password = admin, password = password, password = changeme, "
            "password = 123456, default_password, weak_password, "
            "master_password = admin"
        ),
        "remediation": (
            "Generate strong, random passwords using a secrets manager. "
            "Require passwords to be at least 16 characters with mixed case, "
            "digits, and symbols. Rotate passwords regularly."
        ),
        "references":  "CWE-521, OWASP A07:2021",
    },
    {
        "id":          "general-02",
        "category":    "general",
        "severity":    "MEDIUM",
        "title":       "Insecure HTTP Endpoint Exposed",
        "description": (
            "Services configured to listen on HTTP (port 80) without redirect to "
            "HTTPS expose data in transit and are vulnerable to man-in-the-middle "
            "attacks. CWE-319."
        ),
        "indicators":  (
            "port = 80, listener port 80, http not https, "
            "redirect_to_https = false, ssl_redirect = false"
        ),
        "remediation": (
            "Configure HTTP → HTTPS redirect. Force HTTPS-only listeners. "
            "Use ACM/Let's Encrypt certificates. "
            "Set `redirect_to_https = true` on load balancers."
        ),
        "references":  "CWE-319, OWASP A02:2021",
    },
    {
        "id":          "general-03",
        "category":    "general",
        "severity":    "HIGH",
        "title":       "Deletion Protection Disabled on Critical Resources",
        "description": (
            "Databases, clusters, and stateful resources without deletion protection "
            "can be accidentally or maliciously destroyed, causing data loss. "
        ),
        "indicators":  (
            "deletion_protection = false, skip_final_snapshot = true, "
            "force_destroy = true, prevent_destroy missing, "
            "deletion_protection disabled"
        ),
        "remediation": (
            "Set `deletion_protection = true` on RDS, Aurora, and Elasticsearch "
            "clusters. Use `lifecycle { prevent_destroy = true }` in Terraform "
            "for stateful resources. Take automated snapshots."
        ),
        "references":  "AWS RDS Best Practices, Terraform lifecycle docs",
    },
    {
        "id":          "general-04",
        "category":    "general",
        "severity":    "MEDIUM",
        "title":       "Missing or Disabled Backup Configuration",
        "description": (
            "Resources without backup enabled cannot recover from accidental "
            "deletion, ransomware, or corruption. RDS, DynamoDB, and EFS "
            "support automated backups."
        ),
        "indicators":  (
            "backup_retention_period = 0, backup_enabled = false, "
            "automated_backups disabled, point_in_time_recovery disabled"
        ),
        "remediation": (
            "Set `backup_retention_period >= 7` (preferably 35 days). "
            "Enable `point_in_time_recovery` for DynamoDB. "
            "Use AWS Backup for centralised backup management."
        ),
        "references":  "CIS AWS 2.11, AWS Well-Architected Reliability Pillar",
    },
    {
        "id":          "general-05",
        "category":    "general",
        "severity":    "MEDIUM",
        "title":       "Unrestricted Egress Traffic",
        "description": (
            "Security groups or firewall rules allowing all outbound traffic "
            "(0.0.0.0/0 egress) enable data exfiltration and C2 communication "
            "from compromised instances."
        ),
        "indicators":  (
            "egress cidr 0.0.0.0/0, outbound all traffic allowed, "
            "egress from_port = 0 to_port = 0, allow all egress"
        ),
        "remediation": (
            "Restrict outbound traffic to known destinations and required ports. "
            "Use VPC endpoints for AWS service access. "
            "Implement network segmentation with restrictive egress rules."
        ),
        "references":  "CIS AWS 4.3, CWE-200",
    },
    {
        "id":          "general-06",
        "category":    "general",
        "severity":    "HIGH",
        "title":       "Terraform State Stored Without Encryption or Access Control",
        "description": (
            "Terraform remote state files (S3, Terraform Cloud) may contain "
            "sensitive resource attributes including connection strings, passwords, "
            "and ARNs. Unencrypted or publicly accessible state is a data leak risk."
        ),
        "indicators":  (
            "terraform backend s3 encrypt = false, state file unencrypted, "
            "backend s3 no kms, state publicly readable"
        ),
        "remediation": (
            "Enable S3 server-side encryption for the Terraform state bucket with "
            "`encrypt = true` and a KMS key. Restrict S3 bucket access to the "
            "CI/CD role only. Enable S3 versioning and MFA delete on the state bucket."
        ),
        "references":  "Terraform Security Best Practices, CWE-312",
    },
    {
        "id":          "general-07",
        "category":    "general",
        "severity":    "HIGH",
        "title":       "Missing Network Segmentation (Flat Network)",
        "description": (
            "Placing all resources in a single VPC/subnet without network "
            "segmentation (public vs private subnets, NACLs, security groups) "
            "allows lateral movement once a single resource is compromised."
        ),
        "indicators":  (
            "single subnet, no private subnet, all resources public subnet, "
            "no network segmentation, flat network"
        ),
        "remediation": (
            "Use a multi-tier architecture: public subnets for load balancers, "
            "private subnets for compute, isolated subnets for databases. "
            "Apply NACLs as a secondary control. Use AWS Transit Gateway for "
            "segmentation at scale."
        ),
        "references":  "AWS Well-Architected Security Pillar, CIS AWS 4.1",
    },

    # ── Azure IAM / Managed Identity ───────────────────────────────────────
    {
        "id":          "cis-az-iam-01",
        "category":    "cis_azure",
        "severity":    "CRITICAL",
        "title":       "Azure Managed Identity Assigned Owner or Contributor at Subscription Scope",
        "description": (
            "Assigning the Owner or Contributor role to a managed identity at "
            "subscription scope violates the principle of least privilege. A "
            "compromised identity can create, modify, or delete any resource in "
            "the entire subscription, escalate privileges, and exfiltrate data. "
            "CIS Azure Benchmark 1.23 — Ensure managed identity is not assigned "
            "Owner or Co-Administrator role."
        ),
        "indicators":  (
            "azurerm_role_assignment scope subscription, role_definition_name Owner, "
            "role_definition_name Contributor, subscription scope IAM, "
            "User Access Administrator subscription, role assignment /subscriptions/"
        ),
        "remediation": (
            "Remove subscription-level Owner/Contributor assignments. "
            "Replace with the most restrictive role that fulfils the workload need: "
            "Storage Blob Data Contributor for storage, Virtual Machine Contributor "
            "for VMs, Key Vault Secrets User for secret access. "
            "Scope assignments to the specific resource group or resource. "
            "Regularly audit role assignments using Azure Access Reviews."
        ),
        "references":  "CIS Azure 1.23, CIS Azure 1.1, Azure RBAC Best Practices",
    },
    {
        "id":          "cis-az-iam-02",
        "category":    "cis_azure",
        "severity":    "HIGH",
        "title":       "Azure Contributor Role at Resource Group Scope Without Justification",
        "description": (
            "Contributor at resource group scope allows modification and deletion "
            "of all resources in the group. For managed identities backing single-purpose "
            "workloads (e.g., a function app reading blobs), this is far broader than "
            "necessary. CIS Azure 1.22."
        ),
        "indicators":  (
            "role_definition_name Contributor, scope azurerm_resource_group, "
            "Contributor resource group, azurerm_role_assignment resource_group scope"
        ),
        "remediation": (
            "Identify the specific resources and operations the identity requires. "
            "Replace Contributor with a service-specific built-in role: "
            "  • Storage workloads    → Storage Blob Data Contributor\n"
            "  • VM management        → Virtual Machine Contributor\n"
            "  • Key Vault access     → Key Vault Secrets User\n"
            "  • Network management   → Network Contributor\n"
            "If no built-in role fits, create a custom role with only the required actions."
        ),
        "references":  "CIS Azure 1.22, Azure least privilege documentation",
    },
    {
        "id":          "cis-az-iam-03",
        "category":    "cis_azure",
        "severity":    "HIGH",
        "title":       "Excessive Role Assignments for a Single Managed Identity",
        "description": (
            "A managed identity accumulating many role assignments increases the "
            "blast radius if it is compromised and makes permission auditing "
            "difficult. Each additional role compounds the risk. "
            "CIS Azure 1.21: Ensure that no custom subscription owner roles are created."
        ),
        "indicators":  (
            "multiple azurerm_role_assignment same principal_id, "
            "identity has more than 3 roles, overlapping role assignments, "
            "redundant permissions"
        ),
        "remediation": (
            "Audit all role assignments for each managed identity. "
            "Remove roles that are no longer needed or that overlap with others. "
            "If a single identity genuinely needs broad access, consider splitting "
            "the workload across multiple identities with narrower scopes. "
            "Use Azure Access Reviews to enforce periodic re-validation."
        ),
        "references":  "CIS Azure 1.21, Microsoft Identity Governance",
    },
    {
        "id":          "cis-az-iam-04",
        "category":    "cis_azure",
        "severity":    "HIGH",
        "title":       "User Access Administrator Role Assigned to Managed Identity",
        "description": (
            "User Access Administrator allows management of Azure RBAC assignments "
            "for all resources in scope. A managed identity with this role can grant "
            "itself or other principals any role, enabling full privilege escalation. "
            "This role should never be assigned to automated workloads. "
            "CIS Azure 1.23."
        ),
        "indicators":  (
            "role_definition_name User Access Administrator, "
            "User Access Administrator managed identity, "
            "UAA role assignment terraform"
        ),
        "remediation": (
            "Remove the User Access Administrator assignment from all managed identities. "
            "This role should only be held by human operators under emergency access procedures. "
            "If IAM management automation is required, use a custom role with only the "
            "specific Microsoft.Authorization/roleAssignments/* actions needed, "
            "scoped to a specific resource group."
        ),
        "references":  "CIS Azure 1.23, CWE-269, OWASP A01:2021",
    },
    {
        "id":          "cis-az-iam-05",
        "category":    "cis_azure",
        "severity":    "MEDIUM",
        "title":       "Workload Not Using Managed Identity (Credential-Based Auth)",
        "description": (
            "Using service principal credentials (client secrets or certificates) "
            "instead of managed identities introduces credential rotation burden "
            "and secret-leak risk. Managed identities eliminate the need to store "
            "credentials and are the recommended authentication method for Azure workloads. "
            "Azure Security Benchmark IM-1."
        ),
        "indicators":  (
            "client_secret in terraform, azuread_application_password, "
            "service_principal client_id client_secret, password authentication, "
            "azurerm_key_vault_secret client secret"
        ),
        "remediation": (
            "Replace service principal credential-based authentication with "
            "system-assigned or user-assigned managed identities. "
            "Enable identity { type = SystemAssigned } on the compute resource. "
            "Grant the managed identity the minimum required RBAC role. "
            "For cross-tenant scenarios, use workload identity federation instead "
            "of long-lived secrets."
        ),
        "references":  "Azure Security Benchmark IM-1, CIS Azure 1.20",
    },
    {
        "id":          "cis-az-wi-01",
        "category":    "cis_azure",
        "severity":    "HIGH",
        "title":       "Azure Workload Identity Federation Missing Issuer or Audience Restriction",
        "description": (
            "Federated workload identity credentials must validate the OIDC issuer and pin the "
            "intended audience. Missing issuer validation or absent audience restrictions allows "
            "tokens issued for other workloads or providers to be accepted unexpectedly."
        ),
        "indicators":  (
            "azurerm_federated_identity_credential missing issuer, malformed issuer, audience = [], "
            "audience = [\"*\"], missing aud claim, federated identity credential without audience"
        ),
        "remediation": (
            "Set an explicit HTTPS issuer URL that matches the trusted OIDC provider exactly. "
            "Restrict audiences to the intended value, typically api://AzureADTokenExchange for Azure "
            "workload identity federation, and remove wildcard or empty audiences."
        ),
        "references":  "Azure Workload Identity documentation, OIDC Core 1.0, Azure Security Benchmark IM-3",
    },
    {
        "id":          "cis-az-wi-02",
        "category":    "cis_azure",
        "severity":    "MEDIUM",
        "title":       "OIDC Workload Identity Uses Overly Broad Subject Claims",
        "description": (
            "Wildcard or namespace-wide subject claims weaken federation boundaries by allowing "
            "multiple workloads to impersonate the same Azure identity. Least privilege requires "
            "pinning the subject to the exact Kubernetes service account or external workload subject."
        ),
        "indicators":  (
            "subject = system:serviceaccount:namespace:*, repo:org/repo:*, wildcard subject, "
            "namespace-wide service account trust"
        ),
        "remediation": (
            "Restrict subject to the exact workload, such as system:serviceaccount:<namespace>:<name>. "
            "Avoid wildcard subjects and split identities if multiple workloads need distinct trust."
        ),
        "references":  "CIS Kubernetes Benchmark 5.x, Azure Workload Identity best practices",
    },
    {
        "id":          "cis-az-wi-03",
        "category":    "cis_azure",
        "severity":    "HIGH",
        "title":       "External or Cross-Tenant OIDC Trust Without Strong Restriction",
        "description": (
            "Federating Azure identities with external OIDC issuers or cross-tenant trust boundaries "
            "without explicit subject and tenant restrictions creates a privilege escalation path if the "
            "external identity space is broader than intended."
        ),
        "indicators":  (
            "github actions issuer, token.actions.githubusercontent.com, external oidc issuer, "
            "cross-tenant federation, missing tenant restriction"
        ),
        "remediation": (
            "Use only explicitly approved issuers, pin tenant and subject restrictions, and review "
            "cross-tenant federation separately from in-cluster workload identity trust."
        ),
        "references":  "Azure AD workload identity federation guidance, Microsoft Entra workload identity security",
    },
]


# ---------------------------------------------------------------------------
# SecurityRuleResult
# ---------------------------------------------------------------------------


@dataclass
class SecurityRuleResult:
    """A single security-rule retrieval result."""

    rule_id:   str
    title:     str
    severity:  str
    category:  str
    text:      str      # full rule text used for embedding
    score:     float    # similarity score [0, 1]
    rank:      int

    def to_dict(self) -> dict:
        return {
            "rule_id":  self.rule_id,
            "title":    self.title,
            "severity": self.severity,
            "category": self.category,
            "text":     self.text,
            "score":    self.score,
            "rank":     self.rank,
        }


# ---------------------------------------------------------------------------
# SecurityKnowledgeBase
# ---------------------------------------------------------------------------


class SecurityKnowledgeBase:
    """
    Manages a FAISS-backed index of curated security rules.

    The index is stored separately from the code-chunk index so that
    retrieval for the two corpora can be tuned independently.

    Parameters
    ----------
    embedder :
        An ``EmbeddingGenerator`` instance (shared with the main pipeline
        to avoid loading a second model).
    index_path : str
        Base path (no extension) for the FAISS + SQLite files.
        Defaults to ``"./cache/security_kb"``.
    extra_rules : list[dict], optional
        Additional rule dicts to include alongside the built-in corpus.
        Each dict must have: id, category, severity, title, description,
        indicators, remediation, references.
    """

    def __init__(
        self,
        embedder: Any,
        index_path: str = "./cache/security_kb",
        extra_rules: Optional[List[Dict]] = None,
    ) -> None:
        from vector_store_manager import VectorStoreManager

        self._embedder  = embedder
        self._all_rules = list(_BUILT_IN_RULES) + list(extra_rules or [])

        # Separate VectorStoreManager so rules live in their own index.
        self._store = VectorStoreManager(
            backend="faiss",
            index_path=index_path,
        )
        # Parallel id → rule metadata lookup.
        self._rule_meta: Dict[str, Dict] = {
            r["id"]: r for r in self._all_rules
        }
        # Map numeric index → rule_id so we can reconstruct after search.
        self._index_to_rule_id: List[str] = []

    # ------------------------------------------------------------------
    # Build / persist
    # ------------------------------------------------------------------

    def _rule_to_text(self, rule: Dict) -> str:
        """Concatenate rule fields into a single embeddable document."""
        return (
            f"[{rule['category'].upper()} | {rule['severity']}] "
            f"{rule['title']}\n\n"
            f"{rule['description']}\n\n"
            f"Indicators: {rule['indicators']}\n\n"
            f"Remediation: {rule['remediation']}\n\n"
            f"References: {rule['references']}"
        )

    def build(self) -> int:
        """
        Embed all rules and populate the index.

        Returns the number of rules indexed.
        """
        texts = [self._rule_to_text(r) for r in self._all_rules]
        embeddings = self._embedder.embed(texts)

        # Each rule's chunk_id is its rule id for easy lookup.
        chunks = []
        for i, rule in enumerate(self._all_rules):
            chunks.append({
                "chunk_id":    rule["id"],
                "text":        texts[i],
                "file_path":   f"security_kb/{rule['category']}",
                "file_type":   "security_rule",
                "chunk_index": i,
                "tokens":      len(texts[i].split()),
                "dependencies": [],
                "metadata": {
                    "rule_id":   rule["id"],
                    "category":  rule["category"],
                    "severity":  rule["severity"],
                    "title":     rule["title"],
                },
            })

        self._store.add_embeddings(embeddings, chunks)
        self._store.save_index()
        self._index_to_rule_id = [r["id"] for r in self._all_rules]

        logger.info("SecurityKnowledgeBase built: %d rules indexed", len(self._all_rules))
        return len(self._all_rules)

    def load_or_build(self) -> int:
        """
        Load the saved index if it exists; otherwise build it from scratch.

        Returns the number of rules available.
        """
        if self._store.load():
            n = self._store.total_vectors
            self._index_to_rule_id = [r["id"] for r in self._all_rules[:n]]
            logger.info("SecurityKnowledgeBase loaded: %d rules", n)
            return n
        return self.build()

    def add_rules(self, rules: List[Dict]) -> int:
        """
        Add custom rules to the knowledge base (appended to built-ins).

        *rules* must have the same schema as the built-in rule dicts.
        Returns the number of new rules added.
        """
        texts = [self._rule_to_text(r) for r in rules]
        embeddings = self._embedder.embed(texts)
        chunks = [
            {
                "chunk_id":    r["id"],
                "text":        texts[i],
                "file_path":   f"security_kb/{r.get('category', 'custom')}",
                "file_type":   "security_rule",
                "chunk_index": self._store.total_vectors + i,
                "tokens":      len(texts[i].split()),
                "dependencies": [],
                "metadata": {
                    "rule_id":  r["id"],
                    "category": r.get("category", "custom"),
                    "severity": r.get("severity", "INFO"),
                    "title":    r.get("title", ""),
                },
            }
            for i, r in enumerate(rules)
        ]
        added = self._store.add_embeddings(embeddings, chunks)
        self._all_rules.extend(rules)
        for r in rules:
            self._rule_meta[r["id"]] = r
        self._store.save_index()
        logger.info("SecurityKnowledgeBase: added %d custom rules", added)
        return added

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: "np.ndarray",
        top_k: int = 3,
    ) -> List[SecurityRuleResult]:
        """
        Return the *top_k* most relevant security rules for *query_embedding*.

        Returns an empty list if the index has not been built yet.
        """
        if self._store.total_vectors == 0:
            return []

        raw = self._store.search(query_embedding, top_k=top_k)
        results: List[SecurityRuleResult] = []
        for r in raw:
            meta = r.chunk.get("metadata", {})
            rule_id  = meta.get("rule_id", r.chunk.get("chunk_id", ""))
            rule     = self._rule_meta.get(rule_id, {})
            results.append(
                SecurityRuleResult(
                    rule_id  = rule_id,
                    title    = rule.get("title", meta.get("title", "")),
                    severity = rule.get("severity", meta.get("severity", "INFO")),
                    category = rule.get("category", meta.get("category", "")),
                    text     = r.chunk.get("text", ""),
                    score    = r.score,
                    rank     = r.rank,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_rules(self) -> int:
        """Number of rules currently in the index."""
        return self._store.total_vectors

    @property
    def rule_ids(self) -> List[str]:
        """List of all rule IDs in the built-in corpus."""
        return [r["id"] for r in _BUILT_IN_RULES]
