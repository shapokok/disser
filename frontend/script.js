/**
 * Main JavaScript file for Crop Disease Detection System
 * Common functions and utilities used across pages
 */

// API Configuration
// Automatically detect environment:
// - In production (nginx), use relative URLs to go through proxy
// - In development, connect directly to backend on port 5000
const API_BASE_URL = window.location.port === '80' || window.location.port === ''
    ? window.location.origin  // Use nginx proxy in production
    : 'http://localhost:5000';  // Direct connection in development

// API Helper Functions
const API = {
    /**
     * Upload images to the server
     * @param {FileList} files - Files to upload
     * @returns {Promise} - Upload response
     */
    async uploadImages(files) {
        const formData = new FormData();
        Array.from(files).forEach(file => {
            formData.append('files', file);
        });

        const response = await fetch(`${API_BASE_URL}/api/upload`, {
            method: 'POST',
            body: formData
        });

        return await response.json();
    },

    /**
     * Predict disease from image
     * @param {string} imagePath - Path to uploaded image
     * @param {string} model - Model name
     * @param {string} explanation - Explanation method
     * @param {string} datasetType - Dataset type
     * @returns {Promise} - Prediction response
     */
    async predict(imagePath, model = 'efficientnet', explanation = 'gradcam', datasetType = 'controlled') {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_path: imagePath,
                model: model,
                explanation: explanation,
                dataset_type: datasetType
            })
        });

        return await response.json();
    },

    /**
     * Compare multiple models on same image
     * @param {string} imagePath - Path to uploaded image
     * @param {Array} models - Array of model names
     * @returns {Promise} - Comparison response
     */
    async compareModels(imagePath, models = ['baseline', 'efficientnet', 'mobilenet', 'hybrid']) {
        const response = await fetch(`${API_BASE_URL}/api/compare`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_path: imagePath,
                models: models
            })
        });

        return await response.json();
    },

    /**
     * Batch process multiple images
     * @param {Array} imagePaths - Array of image paths
     * @param {string} model - Model name
     * @param {string} explanation - Explanation method
     * @returns {Promise} - Batch processing response
     */
    async batchPredict(imagePaths, model = 'efficientnet', explanation = 'gradcam') {
        const response = await fetch(`${API_BASE_URL}/api/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_paths: imagePaths,
                model: model,
                explanation: explanation
            })
        });

        return await response.json();
    },

    /**
     * Get model statistics
     * @returns {Promise} - Statistics response
     */
    async getStatistics() {
        const response = await fetch(`${API_BASE_URL}/api/stats`);
        return await response.json();
    },

    /**
     * Get all supported classes
     * @returns {Promise} - Classes response
     */
    async getClasses() {
        const response = await fetch(`${API_BASE_URL}/api/classes`);
        return await response.json();
    },

    /**
     * Get available models
     * @returns {Promise} - Models response
     */
    async getModels() {
        const response = await fetch(`${API_BASE_URL}/api/models`);
        return await response.json();
    },

    /**
     * Check API health
     * @returns {Promise} - Health check response
     */
    async healthCheck() {
        const response = await fetch(`${API_BASE_URL}/`);
        return await response.json();
    }
};

// Utility Functions
const Utils = {
    /**
     * Format file size to human readable format
     * @param {number} bytes - File size in bytes
     * @returns {string} - Formatted size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    },

    /**
     * Format class name for display
     * @param {string} className - Raw class name
     * @returns {string} - Formatted class name
     */
    formatClassName(className) {
        return className.replace(/___/g, ' - ').replace(/_/g, ' ');
    },

    /**
     * Get confidence color class
     * @param {number} confidence - Confidence value (0-1)
     * @returns {string} - CSS class name
     */
    getConfidenceClass(confidence) {
        if (confidence > 0.8) return 'confidence-high';
        if (confidence > 0.5) return 'confidence-medium';
        return 'confidence-low';
    },

    /**
     * Download base64 image
     * @param {string} base64Data - Base64 encoded image
     * @param {string} filename - Filename for download
     */
    downloadBase64Image(base64Data, filename = 'download.png') {
        const link = document.createElement('a');
        link.href = `data:image/png;base64,${base64Data}`;
        link.download = filename;
        link.click();
    },

    /**
     * Show notification message
     * @param {string} message - Message to display
     * @param {string} type - Type of message (success, error, warning, info)
     */
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type}`;
        notification.textContent = message;

        // Add to body
        document.body.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    },

    /**
     * Validate image file
     * @param {File} file - File to validate
     * @returns {Object} - Validation result
     */
    validateImageFile(file) {
        const maxSize = 16 * 1024 * 1024; // 16MB
        const allowedTypes = ['image/jpeg', 'image/png'];

        if (!allowedTypes.includes(file.type)) {
            return {
                valid: false,
                error: 'Invalid file type. Only JPG and PNG are allowed.'
            };
        }

        if (file.size > maxSize) {
            return {
                valid: false,
                error: `File too large. Maximum size is ${Utils.formatFileSize(maxSize)}.`
            };
        }

        return { valid: true };
    },

    /**
     * Create image preview
     * @param {File} file - Image file
     * @returns {Promise<string>} - Data URL
     */
    createImagePreview(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    },

    /**
     * Debounce function
     * @param {Function} func - Function to debounce
     * @param {number} wait - Wait time in ms
     * @returns {Function} - Debounced function
     */
    debounce(func, wait = 300) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Format timestamp
     * @param {string} timestamp - ISO timestamp
     * @returns {string} - Formatted timestamp
     */
    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleString();
    },

    /**
     * Copy text to clipboard
     * @param {string} text - Text to copy
     */
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            Utils.showNotification('Copied to clipboard!', 'success');
        } catch (err) {
            console.error('Failed to copy:', err);
            Utils.showNotification('Failed to copy to clipboard', 'error');
        }
    }
};

// Loading State Manager
const LoadingManager = {
    /**
     * Show loading spinner
     * @param {HTMLElement} container - Container element
     * @param {string} message - Loading message
     */
    show(container, message = 'Loading...') {
        container.innerHTML = `
            <div class="loading-spinner"></div>
            <div class="loading-text">${message}</div>
        `;
        container.classList.remove('hidden');
    },

    /**
     * Hide loading spinner
     * @param {HTMLElement} container - Container element
     */
    hide(container) {
        container.classList.add('hidden');
    }
};

// Error Handler
const ErrorHandler = {
    /**
     * Handle API errors
     * @param {Error} error - Error object
     * @param {HTMLElement} container - Container to display error
     */
    handleAPIError(error, container) {
        console.error('API Error:', error);

        const errorMessage = `
            <div class="alert alert-error">
                <strong>Error:</strong> ${error.message || 'An unexpected error occurred'}
                <br>
                <small>Please ensure the backend server is running on ${API_BASE_URL}</small>
            </div>
        `;

        if (container) {
            container.innerHTML = errorMessage;
            container.classList.remove('hidden');
        }

        Utils.showNotification('An error occurred. Please try again.', 'error');
    }
};

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { API, Utils, LoadingManager, ErrorHandler };
}
