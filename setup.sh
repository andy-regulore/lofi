#!/bin/bash

# LoFi Music Empire - Quick Setup Script
# This script helps you set up the system quickly and easily

set -e  # Exit on error

echo "🎵 LoFi Music Empire - Quick Setup"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "ℹ $1"
}

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3.8+ is required but not found"
    exit 1
fi

# Check if pip is installed
echo ""
echo "Checking pip..."
if command -v pip3 &> /dev/null; then
    print_success "pip3 found"
else
    print_error "pip3 not found. Please install pip first."
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists, skipping..."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
print_info "This may take a few minutes on first run..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
print_success "All dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p output/tracks
mkdir -p output/videos
mkdir -p output/thumbnails
mkdir -p output/sample_packs
mkdir -p data/analytics
mkdir -p logs
print_success "Directories created"

# Check if config.json exists
echo ""
if [ ! -f "config.json" ]; then
    print_warning "config.json not found!"
    echo ""
    print_info "Creating config.json from template..."
    cp config.example.json config.json 2>/dev/null || {
        print_error "config.example.json not found. Please create config.json manually."
        print_info "See GUIDE.md for configuration instructions."
    }
else
    print_success "config.json found"
fi

# Check for API keys
echo ""
echo "Checking configuration..."
if grep -q "YOUR_API_KEY_HERE" config.json 2>/dev/null; then
    print_warning "API keys need to be configured in config.json"
    print_info "Edit config.json and add your YouTube API key"
else
    print_success "Configuration looks good"
fi

# Test imports
echo ""
echo "Testing imports..."
python3 -c "import torch; import transformers; import librosa" 2>/dev/null && {
    print_success "Core libraries import successfully"
} || {
    print_warning "Some imports failed - this might be okay if not using those features"
}

# Success message
echo ""
echo "=================================="
print_success "Setup complete!"
echo "=================================="
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Edit config.json:"
echo "   - Add your YouTube API key (for uploads)"
echo "   - Configure your channel settings"
echo "   - Enable desired features"
echo ""
echo "2. Generate your first track:"
echo "   python src/generate.py --preset study"
echo ""
echo "3. Read the documentation:"
echo "   - GUIDE.md - Complete usage guide"
echo "   - NEW_FEATURES_GUIDE.md - Beginner-friendly explanations"
echo "   - WEBSITE_DEPLOYMENT.md - Deploy your website"
echo ""
echo "4. Deploy the website (optional):"
echo "   - Follow WEBSITE_DEPLOYMENT.md"
echo "   - 30 minutes to live site!"
echo ""
echo "🚀 You're ready to build your LoFi Music Empire!"
echo ""

# Deactivate virtual environment
deactivate 2>/dev/null || true
