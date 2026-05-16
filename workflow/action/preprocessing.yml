name: Telco Customer Churn Preprocessing Pipeline

on:
  push:
    paths:
      - 'preprocessing/**'
      - 'telco-customer-churn_raw/**'
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4
        with:
          persist-credentials: true

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install pandas scikit-learn

      - name: Run preprocessing script
        run: |
          python preprocessing/automate_Jekiwantaufik.py

      - name: Commit results
        run: |
          git config --global user.name "github-actions"
          git config --global user.email "actions@github.com"

          git add preprocessing

          git commit -m "auto update preprocessing dataset" || echo "no changes"

          git push
