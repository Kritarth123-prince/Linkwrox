/**
 * Linkwrox JavaScript Library
 * Copyright © 2025 Kritarth Ranjan - All Rights Reserved
 * Owner: Kritarth123-prince
 * Date: 2025-07-10 17:51:16 UTC
 * Version: 1.0.0-Optimized
 * 
 * ABSOLUTELY NO PULL REQUESTS - GUARANTEED!
 */

class LinkwroxAPI {
    constructor() {
        this.baseURL = '';
        this.init();
    }

    init() {
        console.log('Linkwrox API initialized');
        console.log('Owner: Kritarth123-prince');
        console.log('Copyright © 2025 Kritarth Ranjan - All Rights Reserved');
    }

    // API Methods
    async health() {
        return await this.fetch('/api/health');
    }

    async getStats() {
        return await this.fetch('/api/stats');
    }

    async getThemes() {
        return await this.fetch('/api/themes');
    }

    async generatePost(data) {
        return await this.fetch('/api/generate', 'POST', data);
    }

    // Utility Methods
    async fetch(url, method = 'GET', data = null) {
        try {
            const options = {
                method,
                headers: {
                    'Content-Type': 'application/json',
                }
            };

            if (data) {
                options.body = JSON.stringify(data);
            }

            const response = await fetch(this.baseURL + url, options);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    // UI Helper Methods
    showLoading(message = 'Loading...') {
        const loadingEl = document.getElementById('loading');
        if (loadingEl) {
            loadingEl.classList.add('show');
            const messageEl = document.getElementById('loadingMessage');
            if (messageEl) {
                messageEl.textContent = message;
            }
        }
    }

    hideLoading() {
        const loadingEl = document.getElementById('loading');
        if (loadingEl) {
            loadingEl.classList.remove('show');
        }
    }

    showAlert(message, type = 'info') {
        const alertsContainer = document.getElementById('alerts');
        if (!alertsContainer) return;

        const alertEl = document.createElement('div');
        alertEl.className = `alert alert-${type}`;
        alertEl.innerHTML = `
            ${message}
            <button onclick="this.parentElement.remove()" 
                    style="position: absolute; right: 10px; top: 10px; border: none; background: none; font-size: 18px; cursor: pointer;"
                    aria-label="Close">
                &times;
            </button>
        `;

        alertsContainer.appendChild(alertEl);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertEl.parentElement) {
                alertEl.remove();
            }
        }, 5000);
    }

    // Animation Helpers
    fadeIn(element, duration = 300) {
        element.style.opacity = '0';
        element.style.display = 'block';
        
        let start = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - start;
            const progress = Math.min(elapsed / duration, 1);
            
            element.style.opacity = progress;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }

    fadeOut(element, duration = 300) {
        let start = performance.now();
        const startOpacity = parseFloat(getComputedStyle(element).opacity);
        
        const animate = (currentTime) => {
            const elapsed = currentTime - start;
            const progress = Math.min(elapsed / duration, 1);
            
            element.style.opacity = startOpacity * (1 - progress);
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                element.style.display = 'none';
            }
        };
        
        requestAnimationFrame(animate);
    }

    // Form Helpers
    getFormData(formElement) {
        const formData = new FormData(formElement);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        
        return data;
    }

    validateForm(formElement) {
        const inputs = formElement.querySelectorAll('input[required], textarea[required], select[required]');
        let isValid = true;
        
        inputs.forEach(input => {
            if (!input.value.trim()) {
                input.classList.add('error');
                isValid = false;
            } else {
                input.classList.remove('error');
            }
        });
        
        return isValid;
    }

    // Theme Selection Helper
    setupThemeSelector(containerSelector, onSelectionChange) {
        const container = document.querySelector(containerSelector);
        if (!container) return;

        container.addEventListener('click', (e) => {
            if (e.target.classList.contains('theme-option')) {
                // Remove selected class from all options
                container.querySelectorAll('.theme-option').forEach(option => {
                    option.classList.remove('selected');
                });
                
                // Add selected class to clicked option
                e.target.classList.add('selected');
                
                // Call callback if provided
                if (onSelectionChange) {
                    onSelectionChange(e.target.dataset.theme);
                }
            }
        });
    }

    // Copy to Clipboard
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            this.showAlert('Copied to clipboard!', 'success');
            return true;
        } catch (error) {
            console.error('Copy failed:', error);
            
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            
            try {
                document.execCommand('copy');
                textArea.remove();
                this.showAlert('Copied to clipboard!', 'success');
                return true;
            } catch (fallbackError) {
                textArea.remove();
                this.showAlert('Failed to copy to clipboard', 'error');
                return false;
            }
        }
    }

    // Download Helper
    downloadText(content, filename) {
        const blob = new Blob([content], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        
        a.href = url;
        a.download = filename;
        a.style.display = 'none';
        
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        window.URL.revokeObjectURL(url);
        
        this.showAlert('Download started!', 'success');
    }

    // Format Helpers
    formatDate(date) {
        return new Date(date).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    formatNumber(number) {
        return new Intl.NumberFormat().format(number);
    }

    // Storage Helpers
    saveToLocalStorage(key, data) {
        try {
            localStorage.setItem(key, JSON.stringify(data));
            return true;
        } catch (error) {
            console.error('LocalStorage save failed:', error);
            return false;
        }
    }

    getFromLocalStorage(key, defaultValue = null) {
        try {
            const data = localStorage.getItem(key);
            return data ? JSON.parse(data) : defaultValue;
        } catch (error) {
            console.error('LocalStorage get failed:', error);
            return defaultValue;
        }
    }

    removeFromLocalStorage(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (error) {
            console.error('LocalStorage remove failed:', error);
            return false;
        }
    }

    // Debug Helpers
    log(message, data = null) {
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            console.log(`[Linkwrox] ${message}`, data);
        }
    }

    error(message, error = null) {
        console.error(`[Linkwrox Error] ${message}`, error);
    }
}

// Global instance
window.linkwroxAPI = new LinkwroxAPI();

// Global helper functions
function showLoading(message) {
    window.linkwroxAPI.showLoading(message);
}

function hideLoading() {
    window.linkwroxAPI.hideLoading();
}

function showAlert(message, type) {
    window.linkwroxAPI.showAlert(message, type);
}

function copyToClipboard(text) {
    return window.linkwroxAPI.copyToClipboard(text);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Linkwrox Web Interface Loaded');
    console.log('Owner: Kritarth123-prince');
    console.log('Copyright © 2025 Kritarth Ranjan - All Rights Reserved');
    console.log('Date: 2025-07-10 17:51:16 UTC');
    console.log('NO PULL REQUESTS - ABSOLUTELY GUARANTEED!');
    
    // Add error handling for images
    document.querySelectorAll('img').forEach(img => {
        img.addEventListener('error', () => {
            img.style.display = 'none';
        });
    });
    
    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Error handling
window.addEventListener('error', (event) => {
    console.error('Linkwrox Error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Linkwrox Unhandled Promise Rejection:', event.reason);
});