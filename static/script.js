/* ═══════════════════════════════════════════════════════════════════════════
   AI VIGILANCE — Client Logic
   File: static/script.js
   ═══════════════════════════════════════════════════════════════════════════ */

// ─── Sidebar Toggle ──────────────────────────────────────────────────────
const sidebar = document.getElementById('sidebar');
const mainContent = document.getElementById('main-content');
const toggleBtn = document.getElementById('sb-toggle');

function toggleSidebar() {
    sidebar.classList.toggle('expanded');
    localStorage.setItem('sidebar_expanded', sidebar.classList.contains('expanded'));
}

// Restore sidebar state
if (localStorage.getItem('sidebar_expanded') === 'true') {
    sidebar.classList.add('expanded');
} else {
    sidebar.classList.remove('expanded');
}


// ─── Settings Modal ─────────────────────────────────────────────────────
const settingsModal = document.getElementById('settings-modal');

function openSettings() {
    settingsModal.classList.add('open');
    document.body.classList.add('modal-open');
}

function closeSettings() {
    settingsModal.classList.remove('open');
    document.body.classList.remove('modal-open');
    document.getElementById('settings-msg').textContent = '';
    document.getElementById('settings-form').reset();
}

// Close on outside click
settingsModal.addEventListener('click', (e) => {
    if (e.target === settingsModal) closeSettings();
});
// Close on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && settingsModal.classList.contains('open')) {
        closeSettings();
    }
});

// Update Credentials
document.getElementById('settings-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('save-settings-btn');
    const msg = document.getElementById('settings-msg');
    
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';
    msg.textContent = '';
    
    try {
        const fd = new FormData(e.target);
        const res = await fetch('/api/update_credentials', { method: 'POST', body: fd });
        const data = await res.json();
        
        if (data.status === 'success') {
            msg.className = 'settings-msg success';
            msg.innerHTML = '<i class="fa-solid fa-check-circle"></i> ' + data.message;
            e.target.reset();
            setTimeout(closeSettings, 2000);
        } else {
            msg.className = 'settings-msg error';
            msg.innerHTML = '<i class="fa-solid fa-circle-exclamation"></i> ' + (data.message || 'Update failed');
        }
    } catch(err) {
        msg.className = 'settings-msg error';
        msg.textContent = 'Connection error';
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Save Changes';
    }
});


// ─── Authentication ───────────────────────────────────────────────────────
async function logout() {
    if (!confirm('Are you sure you want to log out?')) return;
    try {
        await fetch('/api/logout', { method: 'POST' });
        window.location.href = '/login';
    } catch(e) { console.error('Logout failed:', e); }
}


// ─── Alert Polling (In-App Only) ──────────────────────────────────────────
const alertPanel = document.getElementById('alert-panel');
const alertBadge = document.getElementById('alert-badge');
const alertList = document.getElementById('alert-list');
let alertPollingInterval;

function toggleAlertPanel() {
    alertPanel.classList.toggle('open');
}

// Close panel if clicked outside
document.addEventListener('click', (e) => {
    const btn = document.getElementById('alert-bell');
    if (!alertPanel.contains(e.target) && !btn.contains(e.target)) {
        alertPanel.classList.remove('open');
    }
});

function formatAlertDate(isoStr) {
    if (!isoStr) return 'Unknown time';
    const d = new Date(isoStr);
    return `${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}:${d.getSeconds().toString().padStart(2,'0')}`;
}

async function fetchAlerts() {
    try {
        const res = await fetch('/api/alerts');
        if (res.status === 401) {
            window.location.href = '/login';
            return;
        }
        const alerts = await res.json();
        const unreadCountResp = await fetch('/api/alerts/unread_count');
        const unreadData = await unreadCountResp.json();
        
        // Update badge
        if (unreadData.count > 0) {
            alertBadge.style.display = 'flex';
            alertBadge.textContent = unreadData.count > 99 ? '99+' : unreadData.count;
            
            // Add pulse animation if newly unread
            alertBadge.animate([
                { transform: 'scale(1)' },
                { transform: 'scale(1.4)' },
                { transform: 'scale(1)' }
            ], { duration: 300, iterations: 1 });
            
        } else {
            alertBadge.style.display = 'none';
        }
        
        // Update panel
        if (!alerts.length) {
            alertList.innerHTML = '<p class="alert-empty"><i class="fa-solid fa-check-double" style="font-size:1.5rem;display:block;margin-bottom:8px;opacity:0.5;"></i>No new alerts</p>';
            return;
        }
        
        alertList.innerHTML = alerts.map(a => `
            <div class="alert-item ${!a.read ? 'unread' : ''}">
                <div class="alert-item-icon">
                    <i class="fa-solid fa-user-shield"></i>
                </div>
                <div class="alert-item-content">
                    <strong>${a.person_name}</strong>
                    <small><i class="fa-solid fa-video" style="font-size:10px;"></i> ${a.camera_id} — ${formatAlertDate(a.timestamp)}</small>
                </div>
            </div>
        `).join('');
        
    } catch(e) { console.error('Alert polling failed', e); }
}

async function markAllRead() {
    try {
        await fetch('/api/alerts/read', { method: 'POST' });
        fetchAlerts();
    } catch(e) { console.error(e); }
}

// Start polling
if (window.location.pathname !== '/login') {
    fetchAlerts();
    alertPollingInterval = setInterval(fetchAlerts, 5000); // Check every 5 seconds
}
