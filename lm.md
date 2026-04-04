# Migrating System Data to VM2

This document outlines the steps to migrate all system data (database, snapshots, recordings, dataset) from the current VM (VM1) to VM2 for centralized storage.

## Prerequisites

- VM1: Current system running AI Vigilance (IP: e.g., 10.7.33.139)
- VM2: Target storage VM (IP: e.g., 10.7.31.184)
- Both VMs should be on the same network
- SSH access configured between VMs
- Sufficient storage space on VM2

## Step 1: Prepare VM2 for Storage

### 1.1 Install Required Packages on VM2
```bash
sudo apt update
sudo apt install sshfs nfs-kernel-server -y
```

### 1.2 Create Storage Directory on VM2
```bash
sudo mkdir -p /data/ai-vigilance
sudo chown -R $USER:$USER /data/ai-vigilance
```

### 1.3 Set up SSH Key Authentication (if not already done)
On VM1:
```bash
ssh-keygen -t rsa -b 4096
ssh-copy-id user@vm2-ip
```

## Step 2: Set up Remote Mounting on VM1

### Option A: Using SSHFS (Recommended for simplicity)

#### 2.1 Install SSHFS on VM1
```bash
sudo apt update
sudo apt install sshfs -y
```

#### 2.2 Create Mount Points on VM1
```bash
sudo mkdir -p /mnt/vm2-data
sudo chown -R $USER:$USER /mnt/vm2-data
```

#### 2.3 Mount VM2 Directory
```bash
sshfs user@vm2-ip:/data/ai-vigilance /mnt/vm2-data
```

#### 2.4 Make Mount Persistent (Optional)
Add to `/etc/fstab`:
```
user@vm2-ip:/data/ai-vigilance /mnt/vm2-data fuse.sshfs defaults,_netdev 0 0
```

### Option B: Using NFS

#### 2.1 Set up NFS on VM2
Edit `/etc/exports`:
```
/data/ai-vigilance vm1-ip(rw,sync,no_subtree_check)
```
Restart NFS:
```bash
sudo systemctl restart nfs-kernel-server
```

#### 2.2 Mount on VM1
```bash
sudo apt install nfs-common -y
sudo mount vm2-ip:/data/ai-vigilance /mnt/vm2-data
```

## Step 3: Migrate Existing Data

### 3.1 Stop the Application
On VM1:
```bash
# Stop any running processes
pkill -f app.py
```

### 3.2 Copy Data to VM2
```bash
# Copy database
cp -r database/ /mnt/vm2-data/

# Copy snapshots
cp -r snapshots/ /mnt/vm2-data/

# Copy recordings
cp -r recordings/ /mnt/vm2-data/

# Copy dataset
cp -r dataset/ /mnt/vm2-data/
```

### 3.3 Verify Data Integrity
```bash
# Check file counts
ls -la /mnt/vm2-data/database/
ls -la /mnt/vm2-data/snapshots/
ls -la /mnt/vm2-data/recordings/
ls -la /mnt/vm2-data/dataset/
```

## Step 4: Update Configuration

### 4.1 Modify core/config.py
Update the directory paths to point to the mounted location:

```python
# ---------------------------------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------------------------------
SNAPSHOT_DIR = "/mnt/vm2-data/snapshots"
DATASET_DIR = "/mnt/vm2-data/dataset"
RECORDING_DIR = "/mnt/vm2-data/recordings"
```

### 4.2 Update Database Path in database/db_manager.py
```python
def __init__(self, db_path='/mnt/vm2-data/database/system.db'):
    self.db_path = db_path
    self.init_db()
```

## Step 5: Test the Setup

### 5.1 Start the Application
```bash
python app.py
```

### 5.2 Verify Data Access
- Check if the web interface loads
- Test camera detection and verify snapshots are saved to VM2
- Check database operations

### 5.3 Monitor Performance
- Ensure network latency doesn't affect real-time processing
- Monitor disk I/O on both VMs

## Step 6: Backup and Maintenance

### 6.1 Set up Automated Backups
Create a cron job for regular backups:
```bash
crontab -e
# Add: 0 2 * * * rsync -av /mnt/vm2-data/ /backup/ai-vigilance/
```

### 6.2 Monitoring
- Set up alerts for mount failures
- Monitor storage usage on VM2

## Troubleshooting

### Mount Issues
- Check network connectivity: `ping vm2-ip`
- Verify SSH keys: `ssh user@vm2-ip`
- Remount: `sudo umount /mnt/vm2-data && sshfs user@vm2-ip:/data/ai-vigilance /mnt/vm2-data`

### Permission Issues
- Ensure user has write permissions on VM2: `chmod 755 /data/ai-vigilance`
- Check mount options

### Performance Issues
- If latency is high, consider local caching
- Use faster network connection between VMs

## Security Considerations

- Use strong SSH keys
- Restrict SSH access to specific IPs
- Encrypt data in transit if sensitive
- Regular security updates on both VMs