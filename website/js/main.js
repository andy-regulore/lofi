// LoFi Music Empire - Main JavaScript
// Handles form submissions, music players, and interactive elements

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Show a success message (appears, then fades out)
 */
function showSuccessMessage(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.add('show');
        // Auto-hide after 5 seconds
        setTimeout(() => {
            element.classList.remove('show');
        }, 5000);
    }
}

/**
 * Validate email format
 */
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

/**
 * Smooth scroll to element
 */
function smoothScroll(targetId) {
    const element = document.getElementById(targetId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// ============================================================================
// CONTACT FORM HANDLER (index.html)
// ============================================================================

if (document.getElementById('contact-form')) {
    document.getElementById('contact-form').addEventListener('submit', function(e) {
        e.preventDefault();

        const name = document.getElementById('name').value;
        const email = document.getElementById('email').value;
        const subject = document.getElementById('subject').value;
        const message = document.getElementById('message').value;

        // Validate email
        if (!isValidEmail(email)) {
            alert('Please enter a valid email address.');
            return;
        }

        // In production, send to your backend API or email service
        // Example endpoint: POST /api/contact
        console.log('Contact form submission:', {
            name,
            email,
            subject,
            message,
            timestamp: new Date().toISOString()
        });

        // Show success message
        showSuccessMessage('contact-success-message');

        // Reset form
        this.reset();

        // Example production code:
        /*
        fetch('/api/contact', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name, email, subject, message })
        })
        .then(response => response.json())
        .then(data => {
            showSuccessMessage('contact-success-message');
            this.reset();
        })
        .catch(error => {
            alert('Error sending message. Please try again or email us directly.');
            console.error('Error:', error);
        });
        */
    });
}

// ============================================================================
// FREE PACK FORM HANDLER (free-pack.html)
// ============================================================================

if (document.getElementById('free-pack-form')) {
    document.getElementById('free-pack-form').addEventListener('submit', function(e) {
        e.preventDefault();

        const name = document.getElementById('name').value;
        const email = document.getElementById('email').value;
        const subscribe = document.getElementById('subscribe').checked;

        // Validate email
        if (!isValidEmail(email)) {
            alert('Please enter a valid email address.');
            return;
        }

        // In production, this would integrate with your email service
        // (Mailchimp, ConvertKit, SendGrid, etc.)
        console.log('Free pack request:', {
            name,
            email,
            subscribe,
            timestamp: new Date().toISOString()
        });

        // Show success message
        showSuccessMessage('success-message');

        // Hide form
        this.style.display = 'none';

        // Example production code (Mailchimp API):
        /*
        fetch('/api/free-pack', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name, email, subscribe })
        })
        .then(response => response.json())
        .then(data => {
            // Send download link via email
            // Add to Mailchimp list if subscribe is true
            showSuccessMessage('success-message');
            this.style.display = 'none';
        })
        .catch(error => {
            alert('Error processing request. Please try again.');
            console.error('Error:', error);
        });
        */

        // Example Mailchimp integration:
        /*
        const mailchimpData = {
            email_address: email,
            status: subscribe ? 'subscribed' : 'transactional',
            merge_fields: {
                FNAME: name.split(' ')[0],
                LNAME: name.split(' ').slice(1).join(' ') || ''
            }
        };

        fetch('YOUR_MAILCHIMP_API_ENDPOINT', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer YOUR_API_KEY'
            },
            body: JSON.stringify(mailchimpData)
        });
        */
    });
}

// ============================================================================
// LICENSING FORM HANDLER (licensing.html)
// ============================================================================

if (document.getElementById('licensing-form')) {
    document.getElementById('licensing-form').addEventListener('submit', function(e) {
        e.preventDefault();

        const email = document.getElementById('email').value;
        const trackName = document.getElementById('track-name').value;
        const licenseType = document.getElementById('license-type').value;

        // Validate email
        if (!isValidEmail(email)) {
            alert('Please enter a valid email address.');
            return;
        }

        // In production, redirect to payment processor or send quote
        console.log('License request:', {
            email,
            trackName,
            licenseType,
            timestamp: new Date().toISOString()
        });

        // Show success message
        showSuccessMessage('license-success-message');

        // Reset form
        this.reset();

        // Example production code (redirect to Stripe checkout):
        /*
        const priceId = {
            'personal': 'price_1234567890',  // Stripe Price ID for $9
            'creator': 'price_0987654321',   // Stripe Price ID for $29
            'commercial': 'price_1122334455' // Stripe Price ID for $99
        }[licenseType];

        fetch('/api/create-checkout-session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                priceId,
                email,
                metadata: { trackName, licenseType }
            })
        })
        .then(response => response.json())
        .then(data => {
            // Redirect to Stripe checkout
            window.location.href = data.checkoutUrl;
        });
        */
    });
}

// ============================================================================
// MUSIC PLAYER CONTROLS
// ============================================================================

/**
 * Initialize all music players on the page
 * This is a placeholder - in production, you'd load actual audio files
 */
function initializeMusicPlayers() {
    const playButtons = document.querySelectorAll('.play-button');

    playButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Toggle play/pause
            const isPlaying = this.textContent === '⏸';

            if (isPlaying) {
                // Pause
                this.textContent = '▶';
                console.log('Paused audio');
                // In production: audioElement.pause();
            } else {
                // Play
                this.textContent = '⏸';
                console.log('Playing audio');

                // Pause all other players
                playButtons.forEach(otherButton => {
                    if (otherButton !== this) {
                        otherButton.textContent = '▶';
                    }
                });

                // In production: load and play actual audio
                /*
                const trackInfo = this.nextElementSibling;
                const trackTitle = trackInfo.querySelector('.track-title').textContent;

                // Create or use existing audio element
                if (!this.audioElement) {
                    this.audioElement = new Audio(`/tracks/${trackTitle}.mp3`);
                    this.audioElement.addEventListener('ended', () => {
                        this.textContent = '▶';
                    });
                }

                this.audioElement.play();
                */
            }
        });
    });
}

// Initialize players when page loads
document.addEventListener('DOMContentLoaded', initializeMusicPlayers);

// ============================================================================
// FADE-IN ANIMATIONS (Scroll-triggered)
// ============================================================================

/**
 * Observe elements and add 'visible' class when they enter viewport
 */
function initializeScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);

    // Observe all fade-in elements
    const fadeElements = document.querySelectorAll('.fade-in');
    fadeElements.forEach(el => observer.observe(el));

    // Observe all cards
    const cards = document.querySelectorAll('.card');
    cards.forEach(el => observer.observe(el));
}

document.addEventListener('DOMContentLoaded', initializeScrollAnimations);

// ============================================================================
// SMOOTH SCROLL FOR NAVIGATION LINKS
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    // Handle smooth scrolling for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');

    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');

            // Skip if href is just "#" or empty
            if (href === '#' || href === '') return;

            const targetId = href.substring(1);
            const targetElement = document.getElementById(targetId);

            if (targetElement) {
                e.preventDefault();
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// ============================================================================
// MOBILE NAVIGATION TOGGLE (Optional Enhancement)
// ============================================================================

/**
 * Add mobile menu toggle for small screens
 * This is a basic implementation - you may want to enhance it
 */
function initializeMobileMenu() {
    const nav = document.querySelector('nav');

    // Create mobile menu button
    const menuButton = document.createElement('button');
    menuButton.className = 'mobile-menu-button';
    menuButton.innerHTML = '☰';
    menuButton.style.display = 'none'; // Hidden by default, show in CSS media query

    // Insert before nav ul
    const navUl = nav.querySelector('ul');
    if (navUl) {
        nav.querySelector('.container').insertBefore(menuButton, navUl);

        menuButton.addEventListener('click', function() {
            navUl.classList.toggle('show');
            this.innerHTML = navUl.classList.contains('show') ? '✕' : '☰';
        });

        // Close menu when clicking a link
        navUl.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navUl.classList.remove('show');
                menuButton.innerHTML = '☰';
            });
        });
    }
}

document.addEventListener('DOMContentLoaded', initializeMobileMenu);

// ============================================================================
// ANALYTICS INTEGRATION (Optional)
// ============================================================================

/**
 * Track user interactions for analytics
 * Integrate with Google Analytics, Plausible, etc.
 */
function trackEvent(category, action, label, value) {
    console.log('Track event:', { category, action, label, value });

    // Example Google Analytics 4 integration:
    /*
    if (typeof gtag !== 'undefined') {
        gtag('event', action, {
            event_category: category,
            event_label: label,
            value: value
        });
    }
    */
}

// Track button clicks
document.addEventListener('DOMContentLoaded', function() {
    // Track CTA button clicks
    const ctaButtons = document.querySelectorAll('.btn-primary, .btn-secondary');
    ctaButtons.forEach(button => {
        button.addEventListener('click', function() {
            trackEvent('CTA', 'click', this.textContent, 1);
        });
    });

    // Track form submissions
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            trackEvent('Form', 'submit', form.id, 1);
        });
    });

    // Track play button clicks
    const playButtons = document.querySelectorAll('.play-button');
    playButtons.forEach(button => {
        button.addEventListener('click', function() {
            const trackTitle = this.nextElementSibling?.querySelector('.track-title')?.textContent || 'Unknown Track';
            trackEvent('Music', 'play', trackTitle, 1);
        });
    });
});

// ============================================================================
// PRODUCTION INTEGRATION NOTES
// ============================================================================

/*
TO MAKE THIS WEBSITE FULLY FUNCTIONAL IN PRODUCTION:

1. EMAIL INTEGRATION:
   - Sign up for Mailchimp, ConvertKit, or SendGrid
   - Get API key
   - Update free-pack form handler to call their API
   - Set up automated email with download link

2. PAYMENT PROCESSING:
   - Set up Stripe or PayPal account
   - Create products/price IDs for each sample pack
   - Add checkout flow to "Buy Now" buttons
   - Set up webhooks for successful payments

3. FILE HOSTING:
   - Upload audio files to cloud storage (AWS S3, Dropbox, etc.)
   - Generate secure, expiring download links
   - Send links via email after payment

4. CONTACT FORM:
   - Set up backend endpoint (Node.js, Python Flask, etc.)
   - Or use service like Formspree, Getform
   - Configure email notifications

5. ANALYTICS:
   - Add Google Analytics 4 or Plausible
   - Uncomment trackEvent() integration
   - Monitor conversions and user behavior

6. AUDIO PLAYERS:
   - Host preview audio files (30-second clips)
   - Update player code to load actual audio
   - Add progress bar and time display

7. SECURITY:
   - Add HTTPS certificate (Let's Encrypt)
   - Implement CSRF protection on forms
   - Rate limit form submissions
   - Validate all inputs server-side

8. SEO:
   - Add sitemap.xml
   - Configure robots.txt
   - Add Open Graph meta tags for social sharing
   - Submit to Google Search Console

ESTIMATED SETUP TIME: 4-6 hours for basic integration
ESTIMATED COST: $0-20/month (depends on hosting and services)
*/

console.log('LoFi Music Empire - Website loaded successfully! 🎵');
