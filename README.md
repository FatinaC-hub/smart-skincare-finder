# Smart Skincare Finder

A semantic search app that ranks skincare products by meaning, not just keywords. Filter by budget, product type and K-Beauty.

**Live demo:** https://smart-skincare-finder.streamlit.app/

![Screenshot](<img width="591" height="582" alt="skincare-finder" src="https://github.com/user-attachments/assets/c3c3b7da-a98f-447a-bfcd-e151581610b1" />)

## How it works
Product descriptions and the user's query are converted to vectors with the all-MiniLM-L6-v2 model. Products are ranked by cosine similarity, then filtered by budget, category and origin.

## Tech stack
Python, Streamlit, sentence-transformers, pandas, NumPy

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Notes
Sample data: prices and descriptions are approximate. Not affiliated with any brand. Built with help from Claude.
