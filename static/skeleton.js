/**
 * BONEYARD SKELETON HELPER — AI Vigilance
 * ----------------------------------------
 * Utility functions to show/hide skeleton screens
 * when dynamic data loads across the app.
 */

const Boneyard = (() => {
    /**
     * Hide a skeleton container and reveal real content.
     * @param {string} skeletonId   - ID of the skeleton element
     * @param {string} [contentId]  - Optional ID of the real content to show
     */
    function hide(skeletonId, contentId) {
        const skel = document.getElementById(skeletonId);
        if (skel) {
            skel.style.transition = 'opacity 0.35s ease';
            skel.style.opacity = '0';
            setTimeout(() => {
                skel.style.display = 'none';
                skel.classList.add('loaded');
            }, 360);
        }
        if (contentId) {
            const content = document.getElementById(contentId);
            if (content) {
                content.style.display = '';
                content.style.opacity = '0';
                requestAnimationFrame(() => {
                    content.style.transition = 'opacity 0.35s ease';
                    content.style.opacity = '1';
                });
            }
        }
    }

    /**
     * Show a skeleton container and hide real content.
     * @param {string} skeletonId
     * @param {string} [contentId]
     */
    function show(skeletonId, contentId) {
        const skel = document.getElementById(skeletonId);
        if (skel) {
            skel.style.display = '';
            skel.style.opacity = '1';
            skel.classList.remove('loaded');
        }
        if (contentId) {
            const content = document.getElementById(contentId);
            if (content) content.style.display = 'none';
        }
    }

    /**
     * Build N repeating skeleton rows for a table tbody.
     * @param {string} tbodyId     - ID of the <tbody>
     * @param {number} cols        - Number of columns
     * @param {number} [rows=5]    - Number of skeleton rows
     */
    function tableRows(tbodyId, cols, rows = 5) {
        const tbody = document.getElementById(tbodyId);
        if (!tbody) return;
        let html = '';
        for (let r = 0; r < rows; r++) {
            html += '<tr>';
            for (let c = 0; c < cols; c++) {
                const w = c === 0 ? '70%' : c === cols - 1 ? '60px' : '50%';
                html += `<td style="padding:12px 15px;">
                    <div class="bone bone-square" style="height:14px;width:${w};"></div>
                </td>`;
            }
            html += '</tr>';
        }
        tbody.innerHTML = html;
    }

    /**
     * Build N repeating feed item skeletons inside a list.
     * @param {string} listId
     * @param {number} [count=4]
     */
    function feedItems(listId, count = 4) {
        const list = document.getElementById(listId);
        if (!list) return;
        let html = '';
        for (let i = 0; i < count; i++) {
            const titleW = [60, 75, 55, 80][i % 4];
            const metaW  = [40, 55, 45, 35][i % 4];
            html += `
            <li class="skeleton-feed-item">
                <div class="bone bone-thumb" style="flex-shrink:0;"></div>
                <div class="skeleton-block" style="flex:1;gap:6px;">
                    <div class="bone bone-square" style="height:14px;width:${titleW}%;"></div>
                    <div class="bone bone-square" style="height:11px;width:${metaW}%;"></div>
                </div>
            </li>`;
        }
        list.innerHTML = html;
    }

    /**
     * Replace a container's content with a chart skeleton.
     * @param {string} containerId
     */
    function chart(containerId) {
        const el = document.getElementById(containerId);
        if (!el) return;
        el.innerHTML = `
        <div class="skeleton-block" style="gap:14px;">
            <div class="skeleton-row" style="justify-content:space-between;">
                <div class="bone bone-square" style="height:14px;width:30%;"></div>
                <div class="bone bone-rounded" style="height:14px;width:15%;"></div>
            </div>
            <div class="bone bone-chart"></div>
        </div>`;
    }

    /**
     * Pulse a stat value element with a bone while loading.
     * Replaces the element inner text with a shimmer bone.
     * @param {string} elementId
     */
    function stat(elementId) {
        const el = document.getElementById(elementId);
        if (!el) return;
        el._origContent = el.innerHTML;
        el.innerHTML = '<div class="bone bone-stat bone-rounded" style="display:inline-block;"></div>';
    }

    /**
     * Restore a stat element to its original content (before bone was applied).
     * @param {string} elementId
     * @param {string} newContent
     */
    function statDone(elementId, newContent) {
        const el = document.getElementById(elementId);
        if (!el) return;
        el.innerHTML = newContent;
    }

    /**
     * Generate skeleton card HTMLs for a grid.
     * @param {number} count
     * @param {'metric'|'camera'|'person'} type
     */
    function gridCards(count, type = 'metric') {
        let html = '';
        for (let i = 0; i < count; i++) {
            if (type === 'metric') {
                html += `
                <div class="skeleton-metric">
                    <div class="bone bone-text-sm" style="width:55%;"></div>
                    <div class="bone bone-stat bone-rounded" style="width:50%;margin-top:4px;"></div>
                    <div class="bone bone-text-sm" style="width:35%;margin-top:2px;"></div>
                </div>`;
            } else if (type === 'camera') {
                html += `
                <div class="skeleton-camera-card">
                    <div class="bone bone-card-img" style="border-radius:0;"></div>
                    <div style="padding:12px;display:flex;flex-direction:column;gap:8px;">
                        <div class="bone bone-text-md" style="width:60%;"></div>
                        <div class="bone bone-text-sm" style="width:40%;"></div>
                        <div style="display:flex;gap:8px;margin-top:4px;">
                            <div class="bone bone-btn"></div>
                            <div class="bone bone-btn"></div>
                        </div>
                    </div>
                </div>`;
            } else if (type === 'person') {
                html += `
                <div class="skeleton-person-card">
                    <div class="bone bone-circle" style="width:64px;height:64px;"></div>
                    <div class="bone bone-text-md" style="width:70%;"></div>
                    <div class="bone bone-badge"></div>
                </div>`;
            }
        }
        return html;
    }

    return { hide, show, tableRows, feedItems, chart, stat, statDone, gridCards };
})();
