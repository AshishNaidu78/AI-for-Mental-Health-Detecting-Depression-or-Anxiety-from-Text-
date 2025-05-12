/**
 * Mental Health Analysis Application
 * Enhanced JavaScript implementation with improved structure, error handling,
 * and modern JavaScript practices
 */

// Use strict mode for better error catching and prevention of unsafe actions
'use strict';

// Main application namespace to avoid global scope pollution
const MentalHealthApp = {
  // Configuration
  config: {
    charLimit: 512,
    statusCheckInterval: 2000,
    statusCheckErrorInterval: 5000
  },
  
  // DOM Elements cache
  elements: {},
  
  /**
   * Initialize the application
   */
  init() {
    this.cacheElements();
    this.bindEvents();
    
    // Initialize specific page functionality based on current page
    if (this.isPage('analysis')) {
      this.initializeAnalysisPage();
    } else if (this.isPage('history')) {
      this.initializeHistoryPage();
    } else if (this.isPage('feedback')) {
      this.initializeFeedbackForm();
    }
    
    // Global functionality
    this.initializeNavigation();
    this.initializeFlashMessages();
    this.checkModelStatus();
  },
  
  /**
   * Cache DOM elements for better performance
   */
  cacheElements() {
    // Navigation elements
    this.elements.navToggle = document.querySelector('.navbar-toggle');
    this.elements.navMenu = document.querySelector('nav ul');
    
    // Text analysis elements
    this.elements.textarea = document.getElementById('text');
    this.elements.charCount = document.querySelector('.char-count');
    this.elements.clearBtn = document.querySelector('.clear-btn');
    
    // Form elements
    this.elements.analysisForm = document.querySelector('.analysis-form form');
    this.elements.submitBtn = this.elements.analysisForm?.querySelector('button[type="submit"]');
    
    // Model selection elements
    this.elements.modelSelect = document.getElementById('model');
    this.elements.stateSelect = document.querySelector('.state-select');
    this.elements.predictionElement = document.querySelector('.result-details h3');
    
    // History page elements
    this.elements.clearHistoryBtn = document.getElementById('clear-history-btn');
    
    // Feedback form elements
    this.elements.feedbackForm = document.querySelector('.feedback-form');
    
    // Loading elements
    this.elements.loadingContainer = document.querySelector('.loading-container');
    this.elements.disabledElements = document.querySelectorAll('[disabled]');
    
    // Flash messages
    this.elements.closeButtons = document.querySelectorAll('.flash-message .close-btn');
  },
  
  /**
   * Bind event listeners
   */
  bindEvents() {
    document.addEventListener('DOMContentLoaded', () => {
      if (this.elements.textarea) {
        this.elements.textarea.addEventListener('input', this.handleTextareaInput.bind(this));
        this.updateCharCount();
      }
      
      if (this.elements.clearBtn) {
        this.elements.clearBtn.addEventListener('click', this.handleClearText.bind(this));
      }
      
      if (this.elements.navToggle) {
        this.elements.navToggle.addEventListener('click', this.toggleNavigation.bind(this));
      }
      
      if (this.elements.closeButtons.length) {
        this.elements.closeButtons.forEach(button => {
          button.addEventListener('click', this.closeFlashMessage);
        });
      }
      
      if (this.elements.modelSelect) {
        this.elements.modelSelect.addEventListener('change', this.handleModelChange.bind(this));
      }
      
      if (this.elements.analysisForm) {
        this.elements.analysisForm.addEventListener('submit', this.handleFormSubmit.bind(this));
      }
      
      if (this.elements.clearHistoryBtn) {
        this.elements.clearHistoryBtn.addEventListener('click', this.handleClearHistory);
      }
      
      if (this.elements.feedbackForm) {
        this.elements.feedbackForm.addEventListener('submit', this.handleFeedbackSubmit);
      }
    });
  },
  
  /**
   * Check if we're on a specific page using CSS classes
   * @param {string} pageName - The page name to check
   * @returns {boolean} - Whether we're on that page
   */
  isPage(pageName) {
    return document.body.classList.contains(`${pageName}-page`);
  },
  
  /**
   * Initialize responsive navigation
   */
  initializeNavigation() {
    // No initialization needed beyond event binding
  },
  
  /**
   * Toggle navigation menu visibility
   */
  toggleNavigation() {
    if (this.elements.navMenu) {
      this.elements.navMenu.classList.toggle('show');
    }
  },
  
  /**
   * Handle textarea input for character counting
   */
  handleTextareaInput() {
    this.updateCharCount();
  },
  
  /**
   * Update character count display
   */
  updateCharCount() {
    if (!this.elements.textarea || !this.elements.charCount) return;
    
    const count = this.elements.textarea.value.length;
    this.elements.charCount.textContent = `${count}/${this.config.charLimit} characters`;
    
    if (count > this.config.charLimit) {
      this.elements.charCount.classList.add('error');
    } else {
      this.elements.charCount.classList.remove('error');
    }
  },
  
  /**
   * Handle clear text button click
   */
  handleClearText() {
    if (this.elements.textarea) {
      this.elements.textarea.value = '';
      this.updateCharCount();
    }
  },
  
  /**
   * Initialize flash message close buttons
   */
  initializeFlashMessages() {
    // No initialization needed beyond event binding
  },
  
  /**
   * Close a flash message
   * @param {Event} e - Click event
   */
  closeFlashMessage(e) {
    e.target.parentElement.remove();
  },
  
  /**
   * Initialize specific analysis page functionality
   */
  initializeAnalysisPage() {
    this.updateStateSelectorVisibility();
  },
  
  /**
   * Update state selector visibility based on prediction
   */
  updateStateSelectorVisibility() {
    if (!this.elements.predictionElement || !this.elements.stateSelect) return;
    
    const prediction = this.elements.predictionElement.textContent.trim();
    this.elements.stateSelect.style.display = prediction === 'Suicidal' ? 'block' : 'none';
  },
  
  /**
   * Handle model selection change
   */
  handleModelChange() {
    if (this.elements.stateSelect) {
      this.elements.stateSelect.style.display = 'none';
    }
  },
  
  /**
   * Check model loading status
   */
  checkModelStatus() {
    if (!this.elements.loadingContainer) return;
    
    const checkStatus = () => {
      fetch('/api/models/status')
        .then(this.handleResponse)
        .then(data => {
          if (data.loaded) {
            this.handleModelsLoaded();
          } else {
            setTimeout(checkStatus, this.config.statusCheckInterval);
          }
        })
        .catch(error => {
          console.error('Error checking model status:', error);
          setTimeout(checkStatus, this.config.statusCheckErrorInterval);
        });
    };
    
    // Start checking
    checkStatus();
  },
  
  /**
   * Handle JSON response and check for errors
   * @param {Response} response - Fetch API response
   * @returns {Promise} - Promise resolving to JSON data
   */
  handleResponse(response) {
    if (!response.ok) {
      throw new Error(`Network error: ${response.status} ${response.statusText}`);
    }
    return response.json();
  },
  
  /**
   * Handle successful model loading
   */
  handleModelsLoaded() {
    if (this.elements.loadingContainer) {
      this.elements.loadingContainer.style.display = 'none';
    }
    
    // Enable all disabled elements
    if (this.elements.disabledElements) {
      this.elements.disabledElements.forEach(element => {
        element.disabled = false;
      });
    }
  },
  
  /**
   * Handle analysis form submission
   * @param {Event} e - Submit event
   */
  handleFormSubmit(e) {
    if (!this.elements.textarea) return;
    
    // Basic validation
    if (!this.elements.textarea.value.trim()) {
      e.preventDefault();
      this.showAlert('Please enter some text for analysis.');
      return false;
    }
    
    // Check character count
    if (this.elements.textarea.value.length > this.config.charLimit) {
      e.preventDefault();
      this.showAlert(`Please limit your text to ${this.config.charLimit} characters.`);
      return false;
    }
    
    // Show loading indicator
    if (this.elements.submitBtn) {
      this.elements.submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
      this.elements.submitBtn.disabled = true;
    }
    
    return true;
  },
  
  /**
   * Initialize the history page functionality
   */
  initializeHistoryPage() {
    // No initialization needed beyond event binding
  },
  
  /**
   * Handle clear history button click
   * @param {Event} e - Click event
   */
  handleClearHistory(e) {
    if (!confirm('Are you sure you want to clear your analysis history?')) {
      e.preventDefault();
    }
  },
  
  /**
   * Initialize feedback form
   */
  initializeFeedbackForm() {
    // No initialization needed beyond event binding
  },
  
  /**
   * Handle feedback form submission
   * @param {Event} e - Submit event
   */
  handleFeedbackSubmit(e) {
    const prediction = document.getElementById('prediction');
    const accurate = document.querySelector('input[name="accurate"]:checked');
    
    if (!prediction?.value) {
      e.preventDefault();
      MentalHealthApp.showAlert('Please select which prediction you are providing feedback on.');
      return false;
    }
    
    if (!accurate) {
      e.preventDefault();
      MentalHealthApp.showAlert('Please indicate whether the prediction was accurate.');
      return false;
    }
    
    return true;
  },
  
  /**
   * Show an alert message
   * @param {string} message - Message to display
   */
  showAlert(message) {
    alert(message); // Could be enhanced to use custom modals
  }
};

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
  MentalHealthApp.init();
});