// Dashboard refresh functionality
let refreshInterval = null;

async function refreshDashboardData() {
    try {
        // Refresh all data
        await Promise.all([
            get_batchs(),
            get_doctors(),
            get_student(),
            get_departments(),
            get_rounds()
        ]);

        // Update tables for visible section
        const activeSection = document.querySelector('.page-section.active');
        if (activeSection) {
            const sectionId = activeSection.id;
            switch (sectionId) {
                case 'dashboard':
                    updateDashboardStats();
                    break;
                case 'students':
                    populateTable('students', students);
                    break;
                case 'doctors':
                    populateTable('doctors', doctors);
                    break;
                case 'departments':
                    populateTable('departments', departments);
                    break;
                case 'Batch':
                    populateTable('Batch', batchs);
                    break;
                case 'rounds':
                    populateTable('rounds', rounds);
                    break;
            }
        }
    } catch (error) {
        console.error('Error refreshing dashboard data:', error);
    }
}

function startPeriodicRefresh() {
    // Clear any existing interval
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }

    // Set up new refresh interval (every 30 seconds)
    refreshInterval = setInterval(async () => {
        await refreshDashboardData();
    }, 30000);
}

// Update the stats display
async function updateDashboardStats() {
    try {
        const totalDoctors = document.getElementById('totalDoctors');
        const totalStudents = document.getElementById('totalStudents');
        const totalDepartments = document.getElementById('totalDepartments');
        const activeRounds = document.getElementById('activeRounds');

        if (totalDoctors) totalDoctors.textContent = doctors.length || 0;
        if (totalStudents) totalStudents.textContent = students.length || 0;
        if (totalDepartments) totalDepartments.textContent = departments.length || 0;
        if (activeRounds) activeRounds.textContent = rounds.length || 0;

        // Add animation class to show update
        [totalDoctors, totalStudents, totalDepartments, activeRounds].forEach(el => {
            if (el) {
                el.classList.add('stat-updated');
                setTimeout(() => el.classList.remove('stat-updated'), 1000);
            }
        });
    } catch (error) {
        console.error('Error updating dashboard stats:', error);
    }
}

// Navigation handler with loading states
async function handlePageNavigation(targetPage) {
    const activeSection = document.getElementById(targetPage);
    if (!activeSection) return;

    // Add loading indicator
    activeSection.classList.add('table-loading');
    const loadingIndicator = document.createElement('div');
    loadingIndicator.className = 'loading-indicator';
    activeSection.appendChild(loadingIndicator);

    try {
        // Refresh data based on section
        switch (targetPage) {
            case 'dashboard':
                await updateDashboardStats();
                break;
            case 'students':
                await get_student();
                break;
            case 'doctors':
                await get_doctors();
                break;
            case 'departments':
                await get_departments();
                break;
            case 'Batch':
                await get_batchs();
                break;
            case 'rounds':
                await get_rounds();
                break;
        }
    } catch (error) {
        console.error('Error refreshing data for page:', targetPage, error);
        showNotification('Error refreshing data', 'error');
    } finally {
        // Remove loading states
        activeSection.classList.remove('table-loading');
        if (loadingIndicator.parentNode) {
            loadingIndicator.parentNode.removeChild(loadingIndicator);
        }
    }
}

// Initialize dashboard functionality
document.addEventListener('DOMContentLoaded', async () => {
    // Initial data load
    await refreshDashboardData();
    
    // Start periodic refresh
    startPeriodicRefresh();

    // Add navigation event listeners
    document.querySelectorAll('.sidebar-icon').forEach(icon => {
        icon.addEventListener('click', async (e) => {
            const targetPage = e.currentTarget.dataset.page;
            if (targetPage) {
                await handlePageNavigation(targetPage);
            }
        });
    });
});

// Add visibility change handler to pause/resume refresh when tab is inactive
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Clear refresh when tab is not visible
        if (refreshInterval) {
            clearInterval(refreshInterval);
            refreshInterval = null;
        }
    } else {
        // Resume refresh when tab becomes visible
        startPeriodicRefresh();
    }
});