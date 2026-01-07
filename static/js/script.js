// Global functions for the retail prediction system
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize charts if on dashboard
    if (document.getElementById('demandChart')) {
        initializeCharts();
    }
    
    // Auto-dismiss alerts after 5 seconds
    autoDismissAlerts();
});

function initializeTooltips() {
    // Add tooltip functionality to elements with data-tooltip attribute
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
    });
}

function showTooltip(event) {
    const tooltipText = event.target.getAttribute('data-tooltip');
    const tooltip = document.createElement('div');
    tooltip.className = 'absolute z-50 px-2 py-1 text-sm text-white bg-gray-900 rounded shadow-lg';
    tooltip.textContent = tooltipText;
    tooltip.id = 'tooltip';
    
    document.body.appendChild(tooltip);
    
    const rect = event.target.getBoundingClientRect();
    tooltip.style.left = `${rect.left + window.scrollX}px`;
    tooltip.style.top = `${rect.top + window.scrollY - tooltip.offsetHeight - 5}px`;
}

function hideTooltip() {
    const tooltip = document.getElementById('tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

function initializeCharts() {
    // Sample demand forecast chart
    const demandCtx = document.getElementById('demandChart').getContext('2d');
    const demandChart = new Chart(demandCtx, {
        type: 'line',
        data: {
            labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
            datasets: [{
                label: 'Demand Forecast',
                data: [65, 59, 80, 81, 56, 55, 70],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Inventory health chart
    const inventoryCtx = document.getElementById('inventoryChart').getContext('2d');
    const inventoryChart = new Chart(inventoryCtx, {
        type: 'doughnut',
        data: {
            labels: ['Optimal', 'Low Risk', 'High Risk'],
            datasets: [{
                data: [60, 25, 15],
                backgroundColor: [
                    '#10b981',
                    '#f59e0b',
                    '#ef4444'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function autoDismissAlerts() {
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            if (alert.parentElement) {
                alert.style.opacity = '0';
                setTimeout(() => alert.remove(), 300);
            }
        }, 5000);
    });
}

// API functions
async function resolveAlert(alertId) {
    try {
        const response = await fetch(`/api/resolve_alert/${alertId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (response.ok) {
            // Remove alert from UI
            const alertElement = document.querySelector(`[data-alert-id="${alertId}"]`);
            if (alertElement) {
                alertElement.style.opacity = '0';
                setTimeout(() => alertElement.remove(), 300);
            }
        } else {
            console.error('Failed to resolve alert');
        }
    } catch (error) {
        console.error('Error resolving alert:', error);
    }
}

// Form validation
function validatePredictionForm() {
    const form = document.getElementById('prediction-form');
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;

    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            field.classList.add('border-red-500');
            isValid = false;
        } else {
            field.classList.remove('border-red-500');
        }
    });

    return isValid;
}

// Loading states
function setLoadingState(button, isLoading) {
    if (isLoading) {
        button.disabled = true;
        button.innerHTML = '<div class="loading"></div> Processing...';
    } else {
        button.disabled = false;
        button.innerHTML = 'Generate Predictions';
    }
}

// Export functionality
function exportData(format, predictionId = null) {
    if (predictionId) {
        // Export single prediction
        window.location.href = `/export/${format}/${predictionId}`;
    } else {
        // Export all predictions
        window.location.href = `/export/all/${format}`;
    }
}

function exportSinglePrediction(format, predictionId) {
    exportData(format, predictionId);
}

function exportAllPredictions(format) {
    if (confirm(`Are you sure you want to export all your predictions as ${format.toUpperCase()}?`)) {
        exportData(format);
    }
}

// Enhanced export with progress indication
function exportWithProgress(format, predictionId = null) {
    const button = event.target;
    const originalText = button.innerHTML;
    
    // Show loading state
    button.innerHTML = '<div class="loading small"></div> Exporting...';
    button.disabled = true;
    
    try {
        if (predictionId) {
            window.location.href = `/export/${format}/${predictionId}`;
        } else {
            window.location.href = `/export/all/${format}`;
        }
    } catch (error) {
        console.error('Export error:', error);
        alert('Error during export. Please try again.');
    } finally {
        // Reset button after a delay
        setTimeout(() => {
            button.innerHTML = originalText;
            button.disabled = false;
        }, 3000);
    }
}

// Download manager
function setupDownloadManager() {
    // Add download event listeners
    document.querySelectorAll('[data-export-format]').forEach(button => {
        button.addEventListener('click', function() {
            const format = this.getAttribute('data-export-format');
            const predictionId = this.getAttribute('data-prediction-id');
            exportWithProgress(format, predictionId);
        });
    });
}

// Initialize download manager when page loads
document.addEventListener('DOMContentLoaded', function() {
    setupDownloadManager();
});

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function convertToCSV(data) {
    // Simplified CSV conversion - you would implement based on your data structure
    const headers = ['Metric', 'Value'];
    const rows = [
        ['Product ID', data.product_info.product_id],
        ['Store ID', data.product_info.store_id],
        ['Demand Forecast', data.demand_forecast.immediate_demand],
        ['Recommended Price', data.pricing_strategy.recommended_price],
        ['Reorder Quantity', data.inventory_management.recommended_order_quantity]
    ];
    
    return [headers, ...rows].map(row => row.join(',')).join('\n');
}