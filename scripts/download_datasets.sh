I #!/bin/bash

# Dataset Download Script
# Downloads and validates datasets for sentiment analysis training
# Following David's recommendations for reproducible data management

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="data/raw"
DATASETS_DIR="datasets"

echo "----------------------------------------------------------------"
echo "  Dataset Download & Validation Script"
echo "----------------------------------------------------------------"
echo ""

# Create directories
mkdir -p "$DATA_DIR/imdb_50k"
mkdir -p "$DATA_DIR/amazon_polarity"
mkdir -p "$DATA_DIR/yelp"

# Function to download file with retry
download_file() {
    local url="$1"
    local output="$2"
    local max_retries=3
    local retry=0

    while [ $retry -lt $max_retries ]; do
        if curl -L -f -o "$output" "$url"; then
            echo -e "${GREEN}[OK]${NC} Downloaded: $(basename "$output")"
            return 0
        else
            retry=$((retry + 1))
            echo -e "${YELLOW}[WARN]${NC} Retry $retry/$max_retries for $(basename "$output")..."
            sleep 2
        fi
    done

    echo -e "${RED}[FAIL]${NC} Failed to download $(basename "$output") after $max_retries attempts"
    return 1
}

# Function to verify checksum
verify_checksum() {
    local file="$1"
    local expected_checksum="$2"

    if [ -z "$expected_checksum" ]; then
        echo -e "${YELLOW}[WARN]${NC} No checksum provided for $(basename "$file"), skipping verification"
        return 0
    fi

    echo "Verifying checksum for $(basename "$file")..."
    if command -v sha256sum &> /dev/null; then
        echo "$expected_checksum  $file" | sha256sum -c -
    elif command -v shasum &> /dev/null; then
        echo "$expected_checksum  $file" | shasum -a 256 -c -
    else
        echo -e "${YELLOW}[WARN]${NC} No checksum utility found, skipping verification"
        return 0
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK]${NC} Checksum verified"
        return 0
    else
        echo -e "${RED}[FAIL]${NC} Checksum verification failed!"
        return 1
    fi
}

echo "----------------------------------------------------------------"
echo "1. IMDB 50K Movie Reviews"
echo "----------------------------------------------------------------"
echo "Source: Stanford AI Lab (Maas et al., 2011)"
echo "URL: https://ai.stanford.edu/~amaas/data/sentiment/"
echo ""

# Check if already downloaded
if [ -f "$DATA_DIR/imdb_50k/IMDB Dataset.csv" ]; then
    echo -e "${GREEN}[OK]${NC} IMDB dataset already exists, skipping download"
else
    echo "Please download manually from:"
    echo "  https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
    echo ""
    echo "Or from the original source:"
    echo "  https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    echo ""
    echo "Save to: $DATA_DIR/imdb_50k/IMDB Dataset.csv"
    echo ""
    echo -e "${YELLOW}Note:${NC} IMDB dataset requires manual download due to source restrictions"
fi

echo ""
echo "----------------------------------------------------------------"
echo "2. Amazon Reviews Polarity"
echo "----------------------------------------------------------------"
echo "Source: Hugging Face (mteb/amazon_polarity)"
echo "URL: https://huggingface.co/datasets/mteb/amazon_polarity"
echo ""

if [ -f "$DATA_DIR/amazon_polarity/train.csv" ]; then
    echo -e "${GREEN}[OK]${NC} Amazon Polarity dataset already exists, skipping download"
else
    echo "Downloading Amazon Polarity dataset..."
    echo ""
    echo -e "${YELLOW}Note:${NC} This dataset is large (3.6M training samples)"
    echo "We'll download a sample or you can use the Hugging Face CLI:"
    echo ""
    echo "  pip install datasets"
    echo "  python -c \"from datasets import load_dataset; ds = load_dataset('mteb/amazon_polarity'); ds['train'].to_csv('$DATA_DIR/amazon_polarity/train.csv'); ds['test'].to_csv('$DATA_DIR/amazon_polarity/test.csv')\""
    echo ""
    echo "For now, creating placeholder instructions..."
    cat > "$DATA_DIR/amazon_polarity/README.txt" << 'EOF'
Amazon Reviews Polarity Dataset Download Instructions

1. Install Hugging Face datasets library:
   pip install datasets

2. Download the dataset using Python:
   python << PYTHON
from datasets import load_dataset

# Load dataset
ds = load_dataset('mteb/amazon_polarity')

# Save to CSV
ds['train'].to_csv('data/raw/amazon_polarity/train.csv', index=False)
ds['test'].to_csv('data/raw/amazon_polarity/test.csv', index=False)

print("[OK] Amazon Polarity dataset downloaded successfully")
PYTHON

3. Alternatively, use the provided Python script:
   python scripts/download_amazon.py

Dataset Info:
- Training: 3,600,000 samples
- Test: 400,000 samples
- Classes: 2 (positive, negative)
- Format: CSV with columns (label, title, text)
EOF
    echo -e "${GREEN}[OK]${NC} Created download instructions at $DATA_DIR/amazon_polarity/README.txt"
fi

echo ""
echo "----------------------------------------------------------------"
echo "3. Yelp Open Dataset"
echo "----------------------------------------------------------------"
echo "Source: Yelp Open Dataset"
echo "URL: https://business.yelp.com/data/resources/open-dataset/"
echo ""

if [ -f "$DATA_DIR/yelp/yelp_academic_dataset_review.json" ]; then
    echo -e "${GREEN}[OK]${NC} Yelp dataset already exists, skipping download"
else
    echo "Please download manually from:"
    echo "  https://business.yelp.com/data/resources/open-dataset/"
    echo ""
    echo "Steps:"
    echo "  1. Visit the URL above and agree to the terms"
    echo "  2. Download the dataset (yelp_dataset.tar)"
    echo "  3. Extract yelp_academic_dataset_review.json"
    echo "  4. Move to: $DATA_DIR/yelp/"
    echo ""
    echo -e "${YELLOW}Note:${NC} Yelp dataset requires agreeing to terms of use"
fi

echo ""
echo "----------------------------------------------------------------"
echo "  Download Summary"
echo "----------------------------------------------------------------"
echo ""

# Check what's downloaded
datasets_ready=0
datasets_total=3

if [ -f "$DATA_DIR/imdb_50k/IMDB Dataset.csv" ]; then
    echo -e "${GREEN}[OK]${NC} IMDB 50K: Ready"
    datasets_ready=$((datasets_ready + 1))
else
    echo -e "${RED}[FAIL]${NC} IMDB 50K: Not downloaded"
fi

if [ -f "$DATA_DIR/amazon_polarity/train.csv" ]; then
    echo -e "${GREEN}[OK]${NC} Amazon Polarity: Ready"
    datasets_ready=$((datasets_ready + 1))
else
    echo -e "${RED}[FAIL]${NC} Amazon Polarity: Not downloaded"
fi

if [ -f "$DATA_DIR/yelp/yelp_academic_dataset_review.json" ]; then
    echo -e "${GREEN}[OK]${NC} Yelp: Ready"
    datasets_ready=$((datasets_ready + 1))
else
    echo -e "${RED}[FAIL]${NC} Yelp: Not downloaded"
fi

echo ""
echo "Datasets ready: $datasets_ready/$datasets_total"
echo ""

if [ $datasets_ready -eq $datasets_total ]; then
    echo -e "${GREEN}All datasets are ready!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run data quality reports: ./scripts/run_quality_reports.sh"
    echo "  2. Preprocess datasets: ./scripts/preprocess_data.sh"
    echo "  3. Train models: ./scripts/train_all_models.sh"
else
    echo -e "${YELLOW}Some datasets are missing.${NC}"
    echo "Please follow the instructions above to download them."
fi

echo ""
echo "----------------------------------------------------------------"
